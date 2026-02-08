/**
 * CHARLTON Python Bindings
 * 
 * pybind11-based Python interface with fluent API
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include "../include/charlton.hpp"

namespace py = pybind11;
using namespace charlton;

// ============================================================================
// Helper: Convert PricingResult to Python dict
// ============================================================================

py::dict pricing_result_to_dict(const PricingResult<double>& r) {
    py::dict d;
    d["price"] = r.price;
    d["delta"] = r.delta;
    d["gamma"] = r.gamma;
    d["theta"] = r.theta;
    d["vega"] = r.vega;
    d["rho"] = r.rho;
    d["vanna"] = r.vanna;
    d["volga"] = r.volga;
    d["zomma"] = r.zomma;
    d["speed"] = r.speed;
    d["charm"] = r.charm;
    d["color"] = r.color;
    d["veta"] = r.veta;
    d["roughness"] = r.roughness;
    d["nu_sens"] = r.nu_sens;
    d["lambda_sens"] = r.lambda_sens;
    d["theta_sens"] = r.theta_sens;
    return d;
}

GreekSet parse_greek_set(const std::string& s) {
    if (s == "price_only") return GreekSet::PRICE_ONLY;
    if (s == "essential") return GreekSet::ESSENTIAL;
    if (s == "standard") return GreekSet::STANDARD;
    if (s == "cornucopia") return GreekSet::CORNUCOPIA;
    return GreekSet::STANDARD;
}

// ============================================================================
// Fluent Interface Helper Classes
// ============================================================================

class FluentModelBuilder {
private:
    RoughHestonPricer<double>::ModelParams params_;
    
public:
    FluentModelBuilder() {
        params_.S0 = 100.0;
        params_.r = 0.05;
        params_.q = 0.0;
        params_.T = 1.0;
        params_.H = 0.1;
        params_.lambda = 2.0;
        params_.theta = 0.04;
        params_.nu = 0.5;
        params_.rho = -0.6;
        params_.V0 = 0.04;
    }
    
    FluentModelBuilder& spot(double S0) { params_.S0 = S0; return *this; }
    FluentModelBuilder& rate(double r) { params_.r = r; return *this; }
    FluentModelBuilder& dividend(double q) { params_.q = q; return *this; }
    FluentModelBuilder& maturity(double T) { params_.T = T; return *this; }
    FluentModelBuilder& hurst(double H) { params_.H = H; return *this; }
    FluentModelBuilder& mean_reversion(double lambda) { params_.lambda = lambda; return *this; }
    FluentModelBuilder& long_term_variance(double theta) { params_.theta = theta; return *this; }
    FluentModelBuilder& vol_of_vol(double nu) { params_.nu = nu; return *this; }
    FluentModelBuilder& correlation(double rho) { params_.rho = rho; return *this; }
    FluentModelBuilder& initial_variance(double V0) { params_.V0 = V0; return *this; }
    
    FluentModelBuilder& rough_heston_standard() {
        params_.H = 0.1;
        params_.lambda = 2.0;
        params_.theta = 0.04;
        params_.nu = 0.5;
        params_.rho = -0.6;
        params_.V0 = 0.04;
        return *this;
    }
    
    FluentModelBuilder& el_euch_rosenbaum() {
        params_.H = 0.12;
        params_.lambda = 0.1;
        params_.theta = 0.3156;
        params_.nu = 0.331;
        params_.rho = -0.681;
        params_.V0 = 0.0392;
        return *this;
    }
    
    FluentModelBuilder& from_atm_iv(double atm_iv) {
        double var = atm_iv * atm_iv;
        params_.theta = var;
        params_.V0 = var;
        return *this;
    }
    
    const RoughHestonPricer<double>::ModelParams& params() const { return params_; }
    
    std::unique_ptr<RoughHestonPricer<double>> pricer(size_t N_time = 256) const {
        return std::make_unique<RoughHestonPricer<double>>(params_, N_time);
    }
};

// ============================================================================
// Fluent Pricer Wrapper
// ============================================================================

class FluentPricer {
private:
    RoughHestonPricer<double>::ModelParams params_;
    
public:
    explicit FluentPricer(const FluentModelBuilder& builder)
        : params_(builder.params()) {}
    
    double put(double K, double error_tol = 1e-10) const {
        RoughHestonPricer<double> pricer(params_);
        return pricer.price_put(K, error_tol);
    }
    
    double call(double K, double error_tol = 1e-10) const {
        RoughHestonPricer<double> pricer(params_);
        return pricer.price_call(K, error_tol);
    }
    
    py::dict greeks_put(double K, const std::string& greek_set = "standard") const {
        GreekSet gset = parse_greek_set(greek_set);
        RoughHestonGreeks<double> greeks(params_);
        auto result = greeks.compute(K, gset);
        return pricing_result_to_dict(result);
    }
    
    py::dict greeks_call(double K, const std::string& greek_set = "standard") const {
        GreekSet gset = parse_greek_set(greek_set);
        RoughHestonGreeks<double> greeks(params_);
        auto result = greeks.compute(K, gset);
        return pricing_result_to_dict(result);
    }
    
    double iv_put(double K) const {
        RoughHestonPricer<double> pricer(params_);
        double price = pricer.price_put(K);
        return RoughHestonPricer<double>::implied_volatility(
            price, params_.S0, K, params_.T, params_.r, false);
    }
    
    double iv_call(double K) const {
        RoughHestonPricer<double> pricer(params_);
        double price = pricer.price_call(K);
        return RoughHestonPricer<double>::implied_volatility(
            price, params_.S0, K, params_.T, params_.r, true);
    }
    
    py::list price_surface(const py::list& strikes, const py::list& maturities, 
                           bool is_call = false) const {
        py::list surface;
        for (auto T_obj : maturities) {
            double T = T_obj.cast<double>();
            auto params_T = params_;
            params_T.T = T;
            RoughHestonPricer<double> pricer_T(params_T);
            
            py::list row;
            for (auto K_obj : strikes) {
                double K = K_obj.cast<double>();
                row.append(is_call ? pricer_T.price_call(K) : pricer_T.price_put(K));
            }
            surface.append(row);
        }
        return surface;
    }
    
    py::dict params() const {
        py::dict d;
        d["S0"] = params_.S0;
        d["r"] = params_.r;
        d["q"] = params_.q;
        d["T"] = params_.T;
        d["H"] = params_.H;
        d["lambda"] = params_.lambda;
        d["theta"] = params_.theta;
        d["nu"] = params_.nu;
        d["rho"] = params_.rho;
        d["V0"] = params_.V0;
        return d;
    }
};

// ============================================================================
// Fluent Calibrator
// ============================================================================

class FluentCalibrator {
private:
    RoughHestonCalibrator<double>::CalibrationParams cal_params_;
    std::unique_ptr<RoughHestonCalibrator<double>> calibrator_;
    
public:
    FluentCalibrator(double S0, double r, double q = 0.0) {
        cal_params_.S0 = S0;
        cal_params_.r = r;
        cal_params_.q = q;
        cal_params_.max_iterations = 1000;
        cal_params_.tolerance = 1e-6;
        cal_params_.step_size = 0.01;
        calibrator_ = std::make_unique<RoughHestonCalibrator<double>>(cal_params_);
    }
    
    FluentCalibrator& max_iterations(int n) { 
        cal_params_.max_iterations = n; 
        calibrator_ = std::make_unique<RoughHestonCalibrator<double>>(cal_params_);
        return *this; 
    }
    
    FluentCalibrator& tolerance(double tol) { 
        cal_params_.tolerance = tol; 
        calibrator_ = std::make_unique<RoughHestonCalibrator<double>>(cal_params_);
        return *this; 
    }
    
    FluentCalibrator& step_size(double s) { 
        cal_params_.step_size = s; 
        calibrator_ = std::make_unique<RoughHestonCalibrator<double>>(cal_params_);
        return *this; 
    }
    
    FluentCalibrator& add_quote(double T, double K, double iv, bool is_call = false) {
        calibrator_->add_quote({T, K, iv, is_call});
        return *this;
    }
    
    FluentCalibrator& add_quotes(const py::list& quotes) {
        for (auto q_obj : quotes) {
            py::tuple q = q_obj.cast<py::tuple>();
            double T = q[0].cast<double>();
            double K = q[1].cast<double>();
            double iv = q[2].cast<double>();
            bool is_call = q.size() > 3 ? q[3].cast<bool>() : false;
            calibrator_->add_quote({T, K, iv, is_call});
        }
        return *this;
    }
    
    py::dict calibrate(const py::dict& initial_guess = py::dict()) {
        RoughHestonCalibrator<double>::CalibrationResult guess;
        
        if (initial_guess.contains("H")) guess.H = initial_guess["H"].cast<double>();
        else guess = calibrator_->generate_initial_guess();
        
        if (initial_guess.contains("lambda")) guess.lambda = initial_guess["lambda"].cast<double>();
        if (initial_guess.contains("theta")) guess.theta = initial_guess["theta"].cast<double>();
        if (initial_guess.contains("nu")) guess.nu = initial_guess["nu"].cast<double>();
        if (initial_guess.contains("rho")) guess.rho = initial_guess["rho"].cast<double>();
        if (initial_guess.contains("V0")) guess.V0 = initial_guess["V0"].cast<double>();
        
        auto result = calibrator_->calibrate_adam(guess);
        
        py::dict d;
        d["H"] = result.H;
        d["lambda"] = result.lambda;
        d["theta"] = result.theta;
        d["nu"] = result.nu;
        d["rho"] = result.rho;
        d["V0"] = result.V0;
        d["rmse"] = result.rmse;
        d["mae"] = result.mae;
        d["iterations"] = result.iterations;
        d["converged"] = result.converged;
        return d;
    }
    
    py::dict generate_initial_guess() const {
        auto guess = calibrator_->generate_initial_guess();
        py::dict d;
        d["H"] = guess.H;
        d["lambda"] = guess.lambda;
        d["theta"] = guess.theta;
        d["nu"] = guess.nu;
        d["rho"] = guess.rho;
        d["V0"] = guess.V0;
        return d;
    }
};

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(_charlton, m) {
    m.doc() = R"doc(
        CHARLTON - Conformal Hyperbolic Accelerated Rough Lévy Transform for Option Numerics
        
        A high-performance library for pricing and calibration in the Rough Heston model.
        
        Quick Start:
            import charlton as ch
            
            # Build a model
            model = ch.model().spot(100).rate(0.05).hurst(0.1).from_atm_iv(0.2)
            
            # Create a pricer
            pricer = ch.pricer(model)
            
            # Price options
            put_price = pricer.put(100)
            call_price = pricer.call(100)
            
            # Get Greeks
            greeks = pricer.greeks_put(100, "cornucopia")
            print(f"Delta: {greeks['delta']}")
    )doc";
    
    // Enums
    py::enum_<GreekSet>(m, "GreekSet")
        .value("PRICE_ONLY", GreekSet::PRICE_ONLY)
        .value("ESSENTIAL", GreekSet::ESSENTIAL)
        .value("STANDARD", GreekSet::STANDARD)
        .value("CORNUCOPIA", GreekSet::CORNUCOPIA);
    
    // Fluent Model Builder
    py::class_<FluentModelBuilder>(m, "ModelBuilder")
        .def(py::init<>())
        .def("spot", &FluentModelBuilder::spot, py::return_value_policy::reference)
        .def("rate", &FluentModelBuilder::rate, py::return_value_policy::reference)
        .def("dividend", &FluentModelBuilder::dividend, py::return_value_policy::reference)
        .def("maturity", &FluentModelBuilder::maturity, py::return_value_policy::reference)
        .def("hurst", &FluentModelBuilder::hurst, py::return_value_policy::reference)
        .def("mean_reversion", &FluentModelBuilder::mean_reversion, py::return_value_policy::reference)
        .def("long_term_variance", &FluentModelBuilder::long_term_variance, py::return_value_policy::reference)
        .def("vol_of_vol", &FluentModelBuilder::vol_of_vol, py::return_value_policy::reference)
        .def("correlation", &FluentModelBuilder::correlation, py::return_value_policy::reference)
        .def("initial_variance", &FluentModelBuilder::initial_variance, py::return_value_policy::reference)
        .def("rough_heston_standard", &FluentModelBuilder::rough_heston_standard, py::return_value_policy::reference)
        .def("el_euch_rosenbaum", &FluentModelBuilder::el_euch_rosenbaum, py::return_value_policy::reference)
        .def("from_atm_iv", &FluentModelBuilder::from_atm_iv, py::return_value_policy::reference)
        .def("params", &FluentModelBuilder::params, py::return_value_policy::reference);
    
    // Fluent Pricer
    py::class_<FluentPricer>(m, "Pricer")
        .def(py::init<const FluentModelBuilder&>())
        .def("put", &FluentPricer::put, py::arg("strike"), py::arg("error_tol") = 1e-10)
        .def("call", &FluentPricer::call, py::arg("strike"), py::arg("error_tol") = 1e-10)
        .def("greeks_put", &FluentPricer::greeks_put, 
             py::arg("strike"), py::arg("greek_set") = "standard")
        .def("greeks_call", &FluentPricer::greeks_call,
             py::arg("strike"), py::arg("greek_set") = "standard")
        .def("iv_put", &FluentPricer::iv_put, py::arg("strike"))
        .def("iv_call", &FluentPricer::iv_call, py::arg("strike"))
        .def("price_surface", &FluentPricer::price_surface,
             py::arg("strikes"), py::arg("maturities"), py::arg("is_call") = false)
        .def("params", &FluentPricer::params);
    
    // Fluent Calibrator
    py::class_<FluentCalibrator>(m, "Calibrator")
        .def(py::init<double, double, double>(),
             py::arg("spot"), py::arg("rate"), py::arg("dividend") = 0.0)
        .def("max_iterations", &FluentCalibrator::max_iterations, py::return_value_policy::reference)
        .def("tolerance", &FluentCalibrator::tolerance, py::return_value_policy::reference)
        .def("step_size", &FluentCalibrator::step_size, py::return_value_policy::reference)
        .def("add_quote", &FluentCalibrator::add_quote, py::return_value_policy::reference)
        .def("add_quotes", &FluentCalibrator::add_quotes, py::return_value_policy::reference)
        .def("calibrate", &FluentCalibrator::calibrate, py::arg("initial_guess") = py::dict())
        .def("generate_initial_guess", &FluentCalibrator::generate_initial_guess);
    
    // Convenience functions
    m.def("model", []() { return FluentModelBuilder(); },
          "Create a new model builder with fluent interface");
    
    m.def("pricer", [](const FluentModelBuilder& model) {
          return FluentPricer(model);
          }, py::arg("model"),
          "Create a pricer from a model builder");
    
    m.def("calibrator", [](double spot, double rate, double dividend) {
          return FluentCalibrator(spot, rate, dividend);
          }, py::arg("spot"), py::arg("rate"), py::arg("dividend") = 0.0,
          "Create a calibrator with fluent interface");
    
    // Utility functions
    m.def("implied_volatility", [](double price, double S0, double K, double T, double r, bool is_call) {
          return RoughHestonPricer<double>::implied_volatility(price, S0, K, T, r, is_call);
          }, py::arg("price"), py::arg("spot"), py::arg("strike"), 
             py::arg("maturity"), py::arg("rate"), py::arg("is_call"),
          "Compute Black-Scholes implied volatility from price");
    
    m.def("benchmark", [](py::function f, int iterations, int warmup) {
          auto cpp_f = [&f]() { f(); };
          return charlton::benchmark(cpp_f, iterations, warmup);
          }, py::arg("func"), py::arg("iterations") = 100, py::arg("warmup") = 10,
          "Benchmark a function (returns average time in ms)");
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "CHARLTON Contributors";
}
