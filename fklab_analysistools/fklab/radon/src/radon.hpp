
#include <utility>
#include <vector>
#include <limits>   
#include <cmath>
#include <stdexcept>

enum class Interpolation { Nearest, Linear };
enum class Constraint { X, Y, No };
enum class IntegralMethod { Integral, Sum, Mean, LogSum, Product};

Constraint constraint_from_string( const std::string & s );
Interpolation interpolation_from_string( const std::string & s );
IntegralMethod integral_from_string( const std::string & s );

typedef std::pair<int, int> IndexPair;
typedef std::pair<IndexPair, IndexPair> IndexLine;

IndexPair indices( unsigned int M, unsigned int N, double alpha, double beta, Interpolation interp  );
double linear_interp_row( double * data, unsigned int stride, int row, double col );
double nearest_interp_row( double * data, unsigned int stride, int row, double col );
double linear_interp_col( double * data, unsigned int stride, double row, int col );
double nearest_interp_col( double * data, unsigned int stride, double row, int col );
double interp_row( Interpolation interp, double * data, unsigned int nrows, unsigned int ncols, int row, double col );
double interp_col( Interpolation interp, double * data, unsigned int nrows, unsigned int ncols, double row, int col );

class Radon {
public:
    Radon( double dx = 1., double dy = 1., Interpolation interpolation = Interpolation::Linear, Constraint constraint = Constraint::No, IntegralMethod integral = IntegralMethod::Sum, bool valid = false, bool intercept = false );
    
    Interpolation interpolation() const;
    Constraint constraint() const;
    IntegralMethod integral_method() const;
    bool valid() const;
    bool intercept() const;
    double dx() const;
    double dy() const;
    
    void set_interpolation( Interpolation val );
    void set_interpolation( std::string val );
    void set_constraint( Constraint val );
    void set_constraint( std::string val );
    void set_integral_method( IntegralMethod val );
    void set_integral_method( std::string val );
    void set_valid( bool val );
    void set_intercept( bool val );
    void set_dx( double val );
    void set_dy( double val );
    
    void transform( double * data, unsigned int nrows, unsigned int ncols,
                    double * theta, unsigned int ntheta,
                    double * rho, unsigned int nrho,
                    double * result, uint16_t * range) const;
    
    void slice( double * data,  unsigned int nrows, unsigned int ncols, 
                double theta, double rho,
                std::vector<double> & result, IndexPair & range ) const;
    
protected:
    double dx_;
    double dy_;
    Interpolation interpolation_;
    Constraint constraint_;
    IntegralMethod integral_method_;
    bool valid_;
    bool intercept_;
};

