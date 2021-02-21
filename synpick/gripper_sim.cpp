
#include <torch/extension.h>

#include <Magnum/Math/Angle.h>

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
using namespace Math::Literals;

namespace
{
    const Matrix4 TIP_POSE = Matrix4::translation({0.0f, 0.0f, -0.055f});

    const Matrix4 SUCTION_JOINT_FRAME = Matrix4::rotationY(90.0_degf);
}

class GripperSim
{
public:
    explicit GripperSim(
        const std::shared_ptr<sl::Scene>& scene,
        const std::shared_ptr<sl::Object>& gripperBase,
        const std::shared_ptr<sl::Object>& gripperCup,
        const Magnum::Matrix4& initialPose
    )
     : m_scene{scene}
     , m_gripperBase{gripperBase}
     , m_gripperCup{gripperCup}
    {
        // Make sure the object is added to the scene
        if(std::find(scene->objects().begin(), scene->objects().end(), m_gripperBase) == scene->objects().end())
            scene->addObject(m_gripperBase);
        if(std::find(scene->objects().begin(), scene->objects().end(), m_gripperCup) == scene->objects().end())
            scene->addObject(m_gripperCup);

        scene->loadPhysics();

        auto& physics = m_scene->context()->physxPhysics();

        // Add a 6D joint which we use to control the manipulator
        m_joint.reset(PxD6JointCreate(physics,
            nullptr, PxTransform{PxIdentity}, // at current pose relative to world (null)
            &m_gripperBase->rigidBody(), PxTransform{PxIdentity} // in origin of manipulator
        ));

        m_joint->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eY, PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
        m_joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
        m_joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLOCKED);
        m_joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLOCKED);

        // Add a 6D joint for the cup
        // The gripper cup rotates around the Y axis.
        // The PhysX joint rotates around the X axis.

        Matrix4 cupJointFrame = Matrix4::rotationZ(90.0_degf);

        m_cupJoint.reset(PxD6JointCreate(physics,
            &m_gripperBase->rigidBody(), PxTransform{cupJointFrame},
            &m_gripperCup->rigidBody(), PxTransform{cupJointFrame}
        ));

        m_cupJoint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
        m_cupJoint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
        m_cupJoint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
        m_cupJoint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLIMITED);
        m_cupJoint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLOCKED);
        m_cupJoint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLOCKED);
        m_cupJoint->setTwistLimit(PxJointAngularLimitPair(-M_PI/2.0, M_PI/2.0));

        m_cupJoint->setDrive(PxD6Drive::eTWIST, PxD6JointDrive(2500.0, 200.0, 200.0));

        // Setup default spring parameters
        setSpringParameters(600.0f, 0.1f, 60.0f);

        // HACK: Disable collisions on the base mesh
        // TODO: Solve this using a simulation callback filter
        std::array<PxShape*, 100> shapes;
        int num = m_gripperBase->rigidBody().getShapes(shapes.data(), shapes.size());
        for(int i = 0; i < num; ++i)
        {
            shapes[i]->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
        }
    }

    ~GripperSim()
    {
        // PhysX is strange.
        m_joint->release();
        m_joint.release();

        m_cupJoint->release();
        m_cupJoint.release();

        // PhysX is strange.
        for(auto& joint : m_graspedObjectJoints)
        {
            joint->release();
            joint.release();
        }
    }

    GripperSim(const GripperSim&) = delete;
    GripperSim& operator=(const GripperSim&) = delete;

    void setSpringParameters(float stiffness, float damping, float forceLimit)
    {
        PxD6JointDrive drive(stiffness, damping, forceLimit);

        m_joint->setDrive(PxD6Drive::eX, drive);
        m_joint->setDrive(PxD6Drive::eY, drive);
        m_joint->setDrive(PxD6Drive::eZ, drive);
    }

    struct IKResult
    {
        Matrix4 basePose;
        Matrix4 cupPose;
        Rad bendAngle;
    };

    IKResult doIK(const Magnum::Vector3& graspPosition, const Magnum::Vector3& graspNormal)
    {
        Vector3 basePosition = graspPosition + graspNormal.normalized() * (-TIP_POSE.translation().z());

        Vector3 baseZ = Vector3::zAxis();
        Vector3 baseX = Vector3{-graspNormal.xy(), 0.0f}.normalized();
        Vector3 baseY = Math::cross(baseZ, baseX).normalized();

        auto baseFrame = Matrix4::from(Matrix3{baseX, baseY, baseZ}, basePosition);

        auto graspInBase = baseFrame.invertedRigid().transformVector(graspNormal);

        Vector3 cupDirection = -graspInBase;

        Rad bendAngle = Rad{std::atan2(-cupDirection.x(), -cupDirection.z())};

        Matrix4 cupPose = baseFrame * Matrix4::rotationY(bendAngle);

        return {
            baseFrame,
            cupPose,
            bendAngle
        };
    }

    void prepareGrasp(const Magnum::Vector3& graspNormal, const Magnum::Vector3& startPosition)
    {
        m_graspNormal = graspNormal;

        IKResult ik = doIK(startPosition, graspNormal);

        m_joint->setLocalPose(PxJointActorIndex::eACTOR0, PxTransform{ik.basePose});
        m_gripperBase->setPose(ik.basePose);
        m_gripperBase->rigidBody().clearForce();
        m_gripperBase->rigidBody().clearTorque();
        m_gripperBase->rigidBody().setLinearVelocity(PxVec3(0.0f, 0.0f, 0.0f));
        m_gripperBase->rigidBody().setAngularVelocity(PxVec3(0.0f, 0.0f, 0.0f));

        m_gripperCup->setPose(ik.cupPose);
        m_gripperCup->rigidBody().clearForce();
        m_gripperCup->rigidBody().clearTorque();
        m_gripperCup->rigidBody().setLinearVelocity(PxVec3(0.0f, 0.0f, 0.0f));
        m_gripperCup->rigidBody().setAngularVelocity(PxVec3(0.0f, 0.0f, 0.0f));

        m_cupJoint->setDrivePosition(PxTransform{Matrix4::rotationX(ik.bendAngle)});

        m_initialPose = ik.basePose;
    }

    void step(const Magnum::Vector3& goalPosition, float dt)
    {
        IKResult ik = doIK(goalPosition, m_graspNormal);

        m_joint->setDrivePosition(PxTransform{m_initialPose.invertedRigid() * ik.basePose});

        auto scene = m_gripperBase->rigidBody().getScene();
        scene->simulate(dt);
        scene->fetchResults(true);

        for(auto& obj : m_scene->objects())
            obj->updateFromPhysics();
    }



    void enableSuction(float goodSealForce, float badSealForce)
    {
        constexpr float MAX_DISTANCE = 0.03;
        constexpr float RADIUS = 0.01f; // 1 cm
        constexpr int NUM_CHECKS = 10;
        constexpr float SWING_LIMIT = 5.0f * M_PI / 180.0f;

        auto scene = m_gripperBase->rigidBody().getScene();
        auto& physics = m_gripperBase->mesh()->context()->physxPhysics();

        PxRaycastBuffer hit;

        std::vector<sl::Object*> objects;

        auto tipPose = m_gripperCup->pose() * TIP_POSE;

        // check if we can make a good seal

        int rayMissed = 0;
        for(int angle_i = 0; angle_i < NUM_CHECKS; angle_i++)
        {
            float angle = angle_i * (2.0f*M_PI/NUM_CHECKS);
            float x = std::cos(angle) * RADIUS;
            float y = std::sin(angle) * RADIUS;
            auto origin = PxVec3{tipPose.transformPoint(Vector3{x, y, -0.001f})};
            auto unitDir = PxVec3{tipPose.transformVector(Vector3{0.f, 0.f, -1.f})};

            scene->raycast(origin, unitDir, MAX_DISTANCE, hit);
            auto hitPositionInWorld = hit.block.position;

            if(!hit.hasBlock)
            {
                rayMissed++;
                continue;
            }

            auto object = reinterpret_cast<sl::Object*>(hit.block.actor->userData);
            if(!object || object->isStatic())
            {
                rayMissed++;
                continue;
            }

            if(object == m_gripperCup.get() || object == m_gripperBase.get())
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
                &m_gripperCup->rigidBody(), PxTransform(TIP_POSE * SUCTION_JOINT_FRAME),
                &obj->rigidBody(), PxTransform(obj->pose().invertedRigid() * tipPose * SUCTION_JOINT_FRAME)
            )};

            joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLIMITED);
            joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLIMITED);

            joint->setLinearLimit(PxJointLinearLimit(0.005, PxSpring(10.0, 0.1)));
            joint->setSwingLimit(PxJointLimitCone(SWING_LIMIT, SWING_LIMIT, PxSpring(10.0, 0.1)));

            joint->setBreakForce(force, force);

            m_graspedObjectJoints.push_back(std::move(joint));
        }
    }

    std::vector<std::shared_ptr<sl::Object>> disableSuction()
    {
        std::vector<std::shared_ptr<sl::Object>> objects;

        auto& sceneObjects = m_scene->objects();

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

            auto it = std::find_if(sceneObjects.begin(), sceneObjects.end(), [&](auto& obj){
                return static_cast<PxRigidActor*>(&obj->rigidBody()) == actor1;
            });
            if(it != sceneObjects.end())
                objects.push_back(*it);
        }

        // PhysX is strange.
        for(auto& joint : m_graspedObjectJoints)
        {
            joint->release();
            joint.release();
        }
        m_graspedObjectJoints.clear();

        return objects;
    }

private:
    std::shared_ptr<sl::Scene> m_scene;
    std::shared_ptr<sl::Object> m_gripperBase;
    std::shared_ptr<sl::Object> m_gripperCup;

    Matrix4 m_initialPose;
    Containers::Pointer<PxD6Joint> m_joint;

    Containers::Pointer<PxD6Joint> m_cupJoint;

    Vector3 m_graspNormal = Vector3::xAxis();

    std::vector<Containers::Pointer<PxD6Joint>> m_graspedObjectJoints;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GripperSim>(m, "GripperSim", R"EOS(
            Simulate a manipulator interacting with the scene.

            The manipulator is driven using a spring damping system (basically Cartesian impedance control).
        )EOS")

        .def(py::init([](const std::shared_ptr<sl::Scene>& scene, const std::shared_ptr<sl::Object>& gripperBase, const std::shared_ptr<sl::Object>& gripperCup, const at::Tensor& initialPose){
                return new GripperSim(
                    scene, gripperBase, gripperCup,
                    magnum::fromTorch<Magnum::Matrix4>::convert(initialPose)
                );
            }), R"EOS(
            Constructor.

            :param scene: The sl.Scene instance to run on
            :param manipulator: The sl.Object to use as manipulator
            :param initial_pose: Initial pose of the manipulator (in world coordinates)
        )EOS", py::arg("scene"), py::arg("gripper_base"), py::arg("gripper_cup"), py::arg("initial_pose"))

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

        .def("prepare_grasp", magnum::wrapRef(&GripperSim::prepareGrasp), R"EOS(
            Reset gripper pose (teleport).

            :param grasp_normal: Normal of the grasped surface
            :param start_position: Start position of the gripper
        )EOS", py::arg("grasp_normal"), py::arg("start_position"))

        .def("enable_suction", &GripperSim::enableSuction, R"EOS(
            Try to suction an object at the current gripper pose.
        )EOS", py::arg("good_seal_force"), py::arg("bad_seal_force"))

        .def("disable_suction", &GripperSim::disableSuction, R"EOS(
            Disable suction.

            :return: A list of objects at the gripper.
        )EOS")
    ;
}
