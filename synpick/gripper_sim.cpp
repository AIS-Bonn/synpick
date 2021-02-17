
#include <torch/extension.h>

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

private:
    std::shared_ptr<sl::Scene> m_scene;
    std::shared_ptr<sl::Object> m_gripper;

    Matrix4 m_initialPose;
    sl::PhysXHolder<PxD6Joint> m_joint;
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
    ;
}
