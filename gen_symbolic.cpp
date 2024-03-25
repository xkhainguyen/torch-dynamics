#include <unistd.h>
#include <sys/stat.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp> // getBase
#include <pinocchio/algorithm/crba.hpp>       // Mass matrix
#include <pinocchio/algorithm/aba.hpp>       // Forward dynamics
#include <pinocchio/algorithm/aba-derivatives.hpp> // Forward dynamics derivative
#include <pinocchio/algorithm/rnea.hpp>       //  Inverse dynamics, nonlinear effects (C)
#include <pinocchio/algorithm/joint-configuration.hpp> // Joint helper
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/multibody/joint/fwd.hpp>

#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/urdf/model.hxx>
#include <pinocchio/parsers/urdf/utils.hpp>
#include <pinocchio/parsers/urdf/geometry.hxx>
#include <urdf_parser/urdf_parser.h>

#include <pinocchio/autodiff/casadi.hpp>

// Casadi typedefs
typedef casadi::SX ADScalar;
typedef casadi::Function ADFunc;
typedef casadi::SXVector ADVector;
typedef casadi::DM DMScalar;
typedef casadi::DMVector DMVector;
typedef casadi::Dict ADDict;
typedef casadi::Slice Slice;

using namespace pinocchio;

// Pinocchio typedefs
using Model = ModelTpl<double, 0, JointCollectionDefaultTpl>;
using ADModel = ModelTpl<ADScalar>;
using Data = DataTpl<double, 0, JointCollectionDefaultTpl>;
using JointModel = JointModelTpl<double_t, 0, JointCollectionDefaultTpl>;

// Joint and foot orderings
std::vector<std::string> joint_names{"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
std::vector<std::string> foot_names{"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
std::vector<std::string> leg_hip_names{"FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"};
std::vector<std::string> leg_thigh_names{"FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"};
std::vector<std::string> leg_calf_names{"FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"};
std::vector<size_t> foot_frame_ids;

GeometryData::SE3 hip_local_frames[4];

// Model and symbolic model
Model model;
Data* data;
pinocchio::ModelTpl<ADScalar> model_sym;
pinocchio::DataTpl<ADScalar> data_sym;

// CasADI sym variables
ADScalar _q, _qdot, _qddot;
ADScalar _tau;

// Pinocchio sym variables
ADModel::ConfigVectorType q_ad, qdot_ad, qddot_ad;
ADModel::TangentVectorType tau_ad;

// Opts
ADDict opts;

// Inputs are q, qdot, and u (tau)
// Outputs are qddot
void gen_forward_dynamics() {
    // Call function and format output
    auto eig_qddot = aba(model_sym, data_sym, q_ad, qdot_ad, tau_ad);
    ADScalar _qddot_out(model.nv, 1);
    for (Eigen::DenseIndex k = 0; k < model.nv; k++) { _qddot_out(k) = eig_qddot(k); }

    // Create symbolic function
    ADFunc eval_aba("eval_forward_dynamics", ADVector{densify(_q), densify(_qdot), densify(_tau)}, ADVector{densify(_qddot_out)});

    // Generate function
    eval_aba.generate("forward_dynamics.c", opts);

    // Generate derivatives (not full-state)
    ADScalar _dqddot_dq = ADScalar::sym("dqddot_dq", model.nv, model.nv);
    ADScalar _dqddot_dqdot = ADScalar::sym("dqddot_dqdot", model.nv, model.nv);
    ADScalar _Minv = ADScalar::sym("Minv", model.nv, model.nv);
    computeABADerivatives(model_sym, data_sym, q_ad, qdot_ad, tau_ad);
    for (int i = 0; i < model.nv; i++) {
        for (int j = 0; j < model.nv; j++) {
            _dqddot_dq(i, j) = data_sym.ddq_dq(i, j);
            _dqddot_dqdot(i, j) = data_sym.ddq_dv(i, j);
            _Minv(i, j) = data_sym.Minv(i, j);
        }
    }

    // Create symbolic function
    ADFunc eval_aba_derivatives("eval_forward_derivatives", ADVector{densify(_q), densify(_qdot), densify(_tau)},
                                ADVector{densify(_dqddot_dq), densify(_dqddot_dqdot), densify(_Minv)});

    // Generate function
    eval_aba_derivatives.generate("forward_derivatives.c", opts);
}

int main(int argc, char* argv[]) {
    std::string rel_path = "cartpole_double.urdf"; 
    std::string current_path(__FILE__);  // Get the full path of the current source file
    std::string abs_path = current_path.substr(0, current_path.find_last_of("/\\") + 1) + rel_path;  
    std::string gen_path = current_path.substr(0, current_path.find_last_of("/\\") + 1) + "/";

    pinocchio::urdf::buildModel(abs_path, model, false);
    model.gravity.linear(model.gravity981);

    // Create data
    data = new Data(model);

    // Define symbolic model
    model_sym = model.cast<ADScalar>();
    data_sym = pinocchio::DataTpl<ADScalar>(model_sym);

    // Define CasADI variables
    _q = ADScalar::sym("q", model.nq);
    _qdot = ADScalar::sym("qdot", model.nq);
    _qddot = ADScalar::sym("qddot", model.nv);
    _tau = ADScalar::sym("tau", model.nv);

    // Define Pinocchio vectors
    q_ad = ADModel::ConfigVectorType(model.nq);
    qdot_ad= ADModel::ConfigVectorType(model.nv);
    qddot_ad = ADModel::TangentVectorType(model.nv);
    tau_ad = ADModel::TangentVectorType(model.nv);

    // Fill Pinocchio vectors with CasADI variables 
    q_ad = Eigen::Map<ADModel::ConfigVectorType>(static_cast<std::vector<ADScalar>>(_q).data(), model.nq, 1);
    qdot_ad = Eigen::Map<ADModel::ConfigVectorType>(static_cast<std::vector<ADScalar>>(_qdot).data(), model.nv, 1);
    qddot_ad = Eigen::Map<ADModel::TangentVectorType>(static_cast<std::vector<ADScalar>>(_qddot).data(), model.nv, 1);
    tau_ad = Eigen::Map<ADModel::TangentVectorType>(static_cast<std::vector<ADScalar>>(_tau).data(), model.nv, 1);

    // Symbolic generation options
    opts.insert(std::pair<std::string, bool>("with_header", true));

    // Change directory so that all generated code is saved
    // to the right place
    int success = chdir(gen_path.c_str());

    // Generate stuff
    gen_forward_dynamics();

}