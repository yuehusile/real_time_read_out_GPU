#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "radon.hpp"

namespace py = pybind11;

PYBIND11_PLUGIN(radonc) {

    py::module m("radonc", "Radon transform");
    
    py::enum_<Interpolation>(m, "Interpolation")
    .value("Linear", Interpolation::Linear)
    .value("Nearest", Interpolation::Nearest)
    .export_values();
    
    py::enum_<Constraint>(m, "Constraint")
    .value("X", Constraint::X)
    .value("Y", Constraint::Y)
    .value("No", Constraint::No)
    .export_values();
    
    py::enum_<IntegralMethod>(m, "IntegralMethod")
    .value("Integral", IntegralMethod::Integral)
    .value("Sum", IntegralMethod::Sum)
    .value("Mean", IntegralMethod::Mean)
    .value("LogSum", IntegralMethod::LogSum)
    .value("Product", IntegralMethod::Product)
    .export_values();
    
    //m.def("indices", &indices );
    
    py::class_<Radon>(m,"Radon")
    .def("__init__", []( Radon & obj, double dx, double dy, const py::object & interp, const py::object & constraint, const py::object & integralmethod, bool valid, bool intercept ) {
        Interpolation _interp;
        Constraint _constraint;
        IntegralMethod _integral;
        
        try { _interp = interpolation_from_string( interp.cast<std::string>() ); }
        catch (std::runtime_error & e ) { _interp = interp.cast<Interpolation>(); }
        
        try { _constraint = constraint_from_string( constraint.cast<std::string>() ); }
        catch (std::runtime_error & e ) { _constraint = constraint.cast<Constraint>(); }
        
        try { _integral = integral_from_string( integralmethod.cast<std::string>() ); }
        catch (std::runtime_error & e ) { _integral = integralmethod.cast<IntegralMethod>(); }
        
        new (&obj) Radon( dx, dy, _interp, _constraint, _integral, valid, intercept);
            
    }, py::arg("dx")=1., py::arg("dy")=1., py::arg("interpolation")=Interpolation::Nearest,
         py::arg("constraint")=Constraint::No, py::arg("integral_method")=IntegralMethod::Sum,
         py::arg("valid")=false, py::arg("intercept")=false )
    .def(py::init<double, double, Interpolation, Constraint, IntegralMethod, bool, bool>(),
         py::arg("dx")=1., py::arg("dy")=1., py::arg("interpolation")=Interpolation::Nearest,
         py::arg("constraint")=Constraint::No, py::arg("integral_method")=IntegralMethod::Sum,
         py::arg("valid")=false, py::arg("intercept")=false)
    .def_property("dx", &Radon::dx, &Radon::set_dx)
    .def_property("dy", &Radon::dy, &Radon::set_dy)
    .def_property("interpolation", &Radon::interpolation, [](Radon & obj, const py::object & value ) {
        try{ 
            obj.set_interpolation( value.cast<std::string>() );
        } catch ( std::runtime_error & e ) {
            obj.set_interpolation( value.cast<Interpolation>() );
        }
    } ) //&Radon::set_interpolation)
    .def_property("constraint", &Radon::constraint, [](Radon & obj, const py::object & value ) {
        try{ 
            obj.set_constraint( value.cast<std::string>() );
        } catch ( std::runtime_error & e ) {
            obj.set_constraint( value.cast<Constraint>() );
        }
    } ) //&Radon::set_constraint)
    .def_property("integral_method", &Radon::integral_method, [](Radon & obj, const py::object & value ) {
        try{ 
            obj.set_integral_method( value.cast<std::string>() );
        } catch ( std::runtime_error & e ) {
            obj.set_integral_method( value.cast<IntegralMethod>() );
        }
    } ) //&Radon::set_integral_method)
    .def_property("valid", &Radon::valid, &Radon::set_valid)
    .def_property("intercept", &Radon::intercept, &Radon::set_intercept)
    .def("slice", []( const Radon & obj, py::array_t<double> data, double theta, double rho ) {
        
        std::vector<double> slice;
        IndexPair n;
        
        // check sample array
        auto buf = data.request();
        
        if (buf.ndim!=2 || buf.shape[0]==0 || buf.shape[1]==0) {
            throw std::runtime_error("Invalid data array.");
        }
        
        obj.slice( (double*) buf.ptr, buf.shape[0], buf.shape[1], theta, rho, slice, n );
        
        return std::make_tuple( slice, n );
        
        }, py::arg("data"), py::arg("theta"), py::arg("rho") )
    .def("transform", []( const Radon & obj, py::array_t<double> data, py::array_t<double> theta, py::array_t<double> rho ) {
        
        // check data array
        auto data_buf = data.request();
        
        if (data_buf.ndim!=2 || data_buf.shape[0]==0 || data_buf.shape[1]==0) {
            throw std::runtime_error("Invalid data array.");
        }
        
        auto theta_buf = theta.request();
        if (theta_buf.ndim!=1 || theta_buf.shape[0]==0) {
            throw std::runtime_error("Invalid theta vector.");
        }
        
        auto rho_buf = rho.request();
        if (rho_buf.ndim!=1 || rho_buf.shape[0]==0) {
            throw std::runtime_error("Invalid rho vector.");
        }
        
        // create output buffer
        auto result = py::array( py::buffer_info(
            nullptr,
            sizeof(double),
            py::format_descriptor<double>::value,
            2,
            { theta_buf.shape[0], rho_buf.shape[0] },
            { sizeof(double) * rho_buf.shape[0], sizeof(double) }
        ));
        
        auto result_buf = result.request();
        
        auto n = py::array( py::buffer_info(
            nullptr,
            sizeof(uint16_t),
            py::format_descriptor<uint16_t>::value,
            3,
            { theta_buf.shape[0], rho_buf.shape[0], 2 },
            { sizeof(uint16_t) * rho_buf.shape[0] * 2, sizeof(uint16_t) * 2, sizeof(uint16_t) }
        ));
        
        auto n_buf = n.request();
        
        obj.transform( (double*) data_buf.ptr, data_buf.shape[0], data_buf.shape[1],
                       (double*) theta_buf.ptr, theta_buf.shape[0],
                       (double*) rho_buf.ptr, rho_buf.shape[0],
                       (double*) result_buf.ptr,
                       (uint16_t*) n_buf.ptr );
        
        return std::make_tuple( result, n );
        
        }, py::arg("data"), py::arg("theta"), py::arg("rho") );
    
    return m.ptr();
    
}
