#include <unistd.h>
#include <sys/stat.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>          // getBase
#include <pinocchio/algorithm/crba.hpp>                // Mass matrix
#include <pinocchio/algorithm/aba.hpp>                 // Forward dynamics
#include <pinocchio/algorithm/aba-derivatives.hpp>     // Forward dynamics derivative
#include <pinocchio/algorithm/rnea.hpp>                //  Inverse dynamics, nonlinear effects (C)
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

// Model and symbolic model
Model model;
Data *data;
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
void gen_cont_forward_dynamics()
{
    // Call function and format output
    auto eig_qddot = aba(model_sym, data_sym, q_ad, qdot_ad, tau_ad);
    ADScalar _qddot_out(model.nv, 1);
    for (Eigen::DenseIndex k = 0; k < model.nv; k++)
    {
        _qddot_out(k) = eig_qddot(k);
    }

    // Create symbolic function
    ADFunc eval_aba("eval_cont_forward_dynamics", ADVector{densify(_q), densify(_qdot), densify(_tau)}, ADVector{densify(_qddot_out)});

    // Generate function
    eval_aba.generate("generated_cont_dynamics.c", opts);

    // Generate derivatives (not full-state)
    ADScalar _dqddot_dq = ADScalar::sym("dqddot_dq", model.nv, model.nv);
    ADScalar _dqddot_dqdot = ADScalar::sym("dqddot_dqdot", model.nv, model.nv);
    ADScalar _Minv = ADScalar::sym("Minv", model.nv, model.nv);
    computeABADerivatives(model_sym, data_sym, q_ad, qdot_ad, tau_ad);
    for (int i = 0; i < model.nv; i++)
    {
        for (int j = 0; j < model.nv; j++)
        {
            _dqddot_dq(i, j) = data_sym.ddq_dq(i, j);
            _dqddot_dqdot(i, j) = data_sym.ddq_dv(i, j);
            _Minv(i, j) = data_sym.Minv(i, j);
        }
    }

    // Create symbolic function
    ADFunc eval_aba_derivatives("eval_cont_forward_derivatives", ADVector{densify(_q), densify(_qdot), densify(_tau)},
                                ADVector{densify(_dqddot_dq), densify(_dqddot_dqdot), densify(_Minv)});

    // Generate function
    eval_aba_derivatives.generate("generated_cont_derivatives.c", opts);
}

void gen_rk4_forward_dynamics()
{
    // Create symbolic var for time
    ADScalar _h = ADScalar::sym("h", 1);

    // Perform rk4 (get points)
    auto qdot1 = qdot_ad;
    auto qddot1 = aba(model_sym, data_sym, q_ad, qdot_ad, tau_ad);
    auto qdot2 = qdot_ad + _h / 2 * qddot1;
    auto qddot2 = aba(model_sym, data_sym, q_ad + _h / 2 * qdot1, qdot_ad + _h / 2 * qddot1, tau_ad);
    auto qdot3 = qdot_ad + _h / 2 * qddot2;
    auto qddot3 = aba(model_sym, data_sym, q_ad + _h / 2 * qdot2, qdot_ad + _h / 2 * qddot2, tau_ad);
    auto qdot4 = qdot_ad + _h * qddot3;
    auto qddot4 = aba(model_sym, data_sym, q_ad + _h * qdot3, qdot_ad + _h * qddot3, tau_ad);

    // Calculate derivatives from candidate points
    auto qdot_rk4 = (qdot1 + 2 * qdot2 + 2 * qdot3 + qdot4) / 6.0;
    auto qddot_rk4 = (qddot1 + 2 * qddot2 + 2 * qddot3 + qddot4) / 6.0;

    // Apply derivatives
    auto q_out = q_ad + _h * qdot_rk4;
    auto qdot_out = qdot_ad + _h * qddot_rk4;

    // Copy over to Casadi types
    ADScalar _q_out = ADScalar::sym("q_out", model.nq);
    ADScalar _qdot_out = ADScalar::sym("qdot_out", model.nq);
    for (int i = 0; i < model.nq; i++)
    {
        _q_out(i) = q_out(i);
        _qdot_out(i) = qdot_out(i);
    }

    // Create symbolic function
    ADFunc eval_rk4("eval_forward_dynamics", ADVector{densify(_q), densify(_qdot), densify(_tau), densify(_h)},
                    ADVector{densify(_q_out), densify(_qdot_out)});

    // Generate function
    eval_rk4.generate("generated_dynamics.c", opts);

    // Generate rk4 derivatives
    ADScalar q_jac_q = ADScalar::jacobian(_q_out, _q);
    ADScalar q_jac_qdot = ADScalar::jacobian(_q_out, _qdot);
    ADScalar q_jac_u = ADScalar::jacobian(_q_out, _tau);
    ADScalar qdot_jac_q = ADScalar::jacobian(_qdot_out, _q);
    ADScalar qdot_jac_qdot = ADScalar::jacobian(_qdot_out, _qdot);
    ADScalar qdot_jac_u = ADScalar::jacobian(_qdot_out, _tau);

    // Create symbolic function
    ADFunc eval_rk4_derivatives("eval_forward_derivatives", ADVector{densify(_q), densify(_qdot), densify(_tau), densify(_h)},
                                ADVector{densify(q_jac_q), densify(q_jac_qdot), densify(q_jac_u),
                                         densify(qdot_jac_q), densify(qdot_jac_qdot), densify(qdot_jac_u)});

    // Generate function
    eval_rk4_derivatives.generate("generated_derivatives.c", opts);
}

int main(int argc, char *argv[])
{
    std::string rel_path = "cartpole1l.urdf";
    std::string current_path(__FILE__); // Get the full path of the current source file
    std::string abs_path = current_path.substr(0, current_path.find_last_of("/\\") + 1) + rel_path;
    std::string gen_path = current_path.substr(0, current_path.find_last_of("/\\") + 1) + "/generated_dynamics/";

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
    qdot_ad = ADModel::ConfigVectorType(model.nv);
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
    gen_cont_forward_dynamics();
    gen_rk4_forward_dynamics();
}