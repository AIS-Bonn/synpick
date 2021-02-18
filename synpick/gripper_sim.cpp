
#include <torch/extension.h>

#include <Corrade/Utility/DebugStl.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>
#include <stillleben/physx.h>
#include <stillleben/physx_impl.h>

#include <stillleben/py_magnum.h>

using namespace Magnum;
using namespace physx;
using namespace sl::python;

class GripperSim
{
public:
    explicit GripperSim(
        const std::shared_ptr<sl::Scene>& scene,
        const std::shared_ptr<sl::Object>& gripper,
        const Magnum::Matrix4& initialPose
    )
     : m_scene{scene}
     , m_gripper{gripper}
     , m_initialPose{initialPose}
    {
        // Make sure the object is added to the scene
        if(std::find(scene->objects().begin(), scene->objects().end(), m_gripper) == scene->objects().end())
            scene->addObject(m_gripper);

        m_gripper->setPose(initialPose);

        scene->loadPhysics();

        auto& physics = m_scene->context()->physxPhysics();

        // Add a 6D joint which we use to control the manipulator
        m_joint.reset(PxD6JointCreate(physics,
            nullptr, PxTransform{initialPose}, // at current pose relative to world (null)
            &m_gripper->rigidBody(), PxTransform{PxIDENTITY::PxIdentity} // in origin of manipulator
        ));

        // By default rotation is locked
        lockRotationAxes(true, true, true);

        // Setup default spring parameters
        setSpringParameters(600.0f, 0.1f, 60.0f);
    }

    GripperSim(const GripperSim&) = delete;
    GripperSim& operator=(const GripperSim&) = delete;

    void lockRotationAxes(bool x, bool y, bool z)
    {
        m_joint->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eY, PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eTWIST, x ? PxD6Motion::eLOCKED : PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eSWING1, y ? PxD6Motion::eLOCKED : PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eSWING2, z ? PxD6Motion::eLOCKED : PxD6Motion::eFREE);
    }

    void setSpringParameters(float stiffness, float damping, float forceLimit)
    {
        PxD6JointDrive drive(stiffness, damping, forceLimit);

        m_joint->setDrive(PxD6Drive::eX, drive);
        m_joint->setDrive(PxD6Drive::eY, drive);
        m_joint->setDrive(PxD6Drive::eZ, drive);
    }

    void step(const Magnum::Matrix4& goalPose, float dt)
    {
        m_joint->setDrivePosition(PxTransform{m_initialPose.invertedRigid() * goalPose});

        auto scene = m_gripper->rigidBody().getScene();
        scene->simulate(dt);
        scene->fetchResults(true);

        for(auto& obj : m_scene->objects())
            obj->updateFromPhysics();
    }

    void resetPoseTo(const Magnum::Matrix4& pose)
    {
        m_joint->setLocalPose(PxJointActorIndex::eACTOR0, PxTransform{pose});
        m_gripper->setPose(pose);
        m_gripper->rigidBody().clearForce();
        m_gripper->rigidBody().clearTorque();
        m_gripper->rigidBody().setLinearVelocity(PxVec3(0.0f, 0.0f, 0.0f));
        m_gripper->rigidBody().setAngularVelocity(PxVec3(0.0f, 0.0f, 0.0f));

        m_initialPose = pose;
    }

    void enableSuction(float goodSealForce, float badSealForce)
    {
        constexpr float MAX_DISTANCE = 0.03;
        constexpr float RADIUS = 0.01f; // 1 cm
        constexpr int NUM_CHECKS = 10;
        constexpr float SWING_LIMIT = 5.0f * M_PI / 180.0f;

        auto scene = m_gripper->rigidBody().getScene();
        auto& physics = m_gripper->mesh()->context()->physxPhysics();

        PxRaycastBuffer hit;

        std::vector<sl::Object*> objects;

        // check if we can make a good seal

        int rayMissed = 0;
        for(int angle_i = 0; angle_i < NUM_CHECKS; angle_i++)
        {
            float angle = angle_i * (2.0f*M_PI/NUM_CHECKS);
            float x = std::cos(angle) * RADIUS;
            float y = std::sin(angle) * RADIUS;
            auto origin = PxVec3{m_gripper->pose().transformPoint(Vector3{x, y, -0.001f})};
            auto unitDir = PxVec3{m_gripper->pose().transformVector(Vector3{0.f, 0.f, -1.f})};

            scene->raycast(origin, unitDir, MAX_DISTANCE, hit);
            auto hitPositionInWorld = hit.block.position;

            auto object = reinterpret_cast<sl::Object*>(hit.block.actor->userData);
            if(!hit.hasBlock || !object || object->isStatic())
                rayMissed++;

            if(object == m_gripper.get())
                throw std::runtime_error{"Hit myself with raycast, that should not happen"};

            auto it = std::find(objects.begin(), objects.end(), object);
            if(it == objects.end())
                objects.push_back(object);
        }

        Debug{} << "Out of" << NUM_CHECKS << "raycasts," << rayMissed << "missed.";

        if(objects.empty())
        {
            Debug{} << "Did not find any objects at suction cup";
            return;
        }

        float force = (rayMissed >= 3) ? badSealForce : goodSealForce;

        Debug{} << "Picking the following objects:";
        for(auto& obj : objects)
        {
            Debug{} << " -" << obj->mesh()->filename();

            Containers::Pointer<PxD6Joint> joint{PxD6JointCreate(
                physics,
                &m_gripper->rigidBody(), PxTransform(Matrix4{}),
                &obj->rigidBody(), PxTransform(obj->pose().invertedRigid() * m_gripper->pose())
            )};

            joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLIMITED);
            joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLIMITED);

            joint->setSwingLimit(PxJointLimitCone(SWING_LIMIT, SWING_LIMIT));

            joint->setBreakForce(force, force);

            m_graspedObjectJoints.push_back(std::move(joint));
        }
    }

    std::vector<std::shared_ptr<sl::Object>> disableSuction()
    {
        std::vector<std::shared_ptr<sl::Object>> objects;

        for(auto& joint : m_graspedObjectJoints)
        {
            if(joint->getConstraintFlags() & PxConstraintFlag::eBROKEN)
            {
                Debug{} << "lost an object on the way";
                continue;
            }

            PxRigidActor* actor0{};
            PxRigidActor* actor1{};
            joint->getActors(actor0, actor1);

            for(auto& object : m_scene->objects())
            {
                if(static_cast<PxRigidActor*>(&object->rigidBody()) == actor1)
                    objects.push_back(object);
            }
        }

        m_graspedObjectJoints.clear();

        return objects;
    }

private:
    std::shared_ptr<sl::Scene> m_scene;
    std::shared_ptr<sl::Object> m_gripper;

    Matrix4 m_initialPose;
    Containers::Pointer<PxD6Joint> m_joint;

    std::vector<Containers::Pointer<PxD6Joint>> m_graspedObjectJoints;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GripperSim>(m, "GripperSim", R"EOS(
            Simulate a manipulator interacting with the scene.

            The manipulator is driven using a spring damping system (basically Cartesian impedance control).
        )EOS")

        .def(py::init([](const std::shared_ptr<sl::Scene>& scene, const std::shared_ptr<sl::Object>& gripper, const at::Tensor& initialPose){
                return new GripperSim(
                    scene, gripper,
                    magnum::fromTorch<Magnum::Matrix4>::convert(initialPose)
                );
            }), R"EOS(
            Constructor.

            :param scene: The sl.Scene instance to run on
            :param manipulator: The sl.Object to use as manipulator
            :param initial_pose: Initial pose of the manipulator (in world coordinates)
        )EOS", py::arg("scene"), py::arg("manipulator"), py::arg("initial_pose"))

        .def("set_spring_parameters", &GripperSim::setSpringParameters, R"EOS(
            Set spring parameters for driving the manipulator.

            :param stiffness: Spring stiffness in N/m
            :param damping: Spring damping in Ns/m
            :param force_limit: Limit of the applied force in N
        )EOS", py::arg("stiffness"), py::arg("damping"), py::arg("force_limit"))

        .def("step", magnum::wrapRef(&GripperSim::step), R"EOS(
            Simulate a single step.

            Take care to take a small enough dt, otherwise the simulation will get unstable.

            :param goal_pose: Goal pose to drive the manipulator to
            :param dt: Step length in s
        )EOS", py::arg("goal_pose"), py::arg("dt"))

        .def("reset_pose_to", magnum::wrapRef(&GripperSim::resetPoseTo), R"EOS(
            Reset gripper pose (teleport).

            :param pose: Pose
        )EOS", py::arg("pose"))

        .def("enable_suction", &GripperSim::enableSuction, R"EOS(
            Try to suction an object at the current gripper pose.
        )EOS", py::arg("good_seal_force"), py::arg("bad_seal_force"))

        .def("disable_suction", &GripperSim::disableSuction, R"EOS(
            Disable suction.

            :return: A list of objects at the gripper.
        )EOS")
    ;
}
