
#include "radon.hpp"
#include <iostream>

Constraint constraint_from_string( const std::string & s ) {
    Constraint ret;
    if (s=="x" || s=="X") {
        ret = Constraint::X;
    } else if (s=="y" || s=="Y") {
        ret = Constraint::Y;
    } else if (s.empty() || s=="none" || s=="None" || s=="N" || s=="n") {
        ret = Constraint::No;
    } else {
        throw std::runtime_error("Cannot convert string to Constraint.");
    }
    
    return ret;
}

Interpolation interpolation_from_string( const std::string & s ) {
    Interpolation ret;
    if (s=="n" || s=="N" || s=="nearest" || s=="Nearest") {
        ret = Interpolation::Nearest;
    } else if (s=="l" || s=="L" || s=="linear" || s=="Linear") {
        ret = Interpolation::Linear;
    } else {
        throw std::runtime_error("Cannot convert string to Interpolation.");
    }
    
    return ret;
}

IntegralMethod integral_from_string( const std::string & s ) {
    IntegralMethod ret;
    if (s=="integral" || s=="Integral") {
        ret = IntegralMethod::Integral;
    } else if (s=="sum" || s=="Sum") {
        ret = IntegralMethod::Sum;
    } else if (s=="mean" || s=="Mean") {
        ret = IntegralMethod::Mean;
    } else if (s=="logsum" || s=="LogSum") {
        ret = IntegralMethod::LogSum;
    } else if (s=="product" || s=="Product") {
        ret = IntegralMethod::Product;
    } else {
        throw std::runtime_error("Cannot convert string to IntegralMethod.");
    }
    
    return ret;
}


IndexPair indices( unsigned int M, unsigned int N, double alpha, double beta, Interpolation interp ) {
    
    if (interp==Interpolation::Nearest) { beta += 0.5; }
    
    IndexPair p = { 0, M };
    
    double eps = std::numeric_limits<double>::epsilon();
    
    if (alpha>std::numeric_limits<double>::epsilon()) {
        p.first = std::ceil( -(beta-eps)/alpha );
        p.second = 1 + std::floor( (N-1-beta-eps)/alpha );
    } else if (alpha<-std::numeric_limits<double>::epsilon()) {
        p.first = std::ceil( (N-1-beta-eps)/alpha );
        p.second = 1 + std::floor( -(beta-eps)/alpha );
    } else if ( (beta-eps)<=0 || (beta+eps)>=(N-1)) {
        p.second = -1;
    }
    
    
    if (p.first<0) { p.first=0; }
    if (p.second> static_cast<int>(M)) { p.second=M; }
    
    return p;
    
}

double linear_interp_row( double * data, unsigned int nrows, unsigned int ncols, int row, double col ) {

    int idx = std::floor( col );
    double w = col - idx;
    
    //if (idx<0 || idx>ncols-1) {
    //    std::cout << "row " << idx << " " << w << std::endl;
    //}
    
    double val;
    
    if (col<0 || col>=ncols-1) {
        val = std::numeric_limits<double>::quiet_NaN();
    //} else if (idx==-1) {
    //    val = data[row * ncols + idx + 1]*w;
    //} else if (idx==static_cast<int>(ncols)-1) {
    //    val = data[row * ncols + idx]*(1-w);
    } else if (std::abs(w) < std::numeric_limits<double>::epsilon()) {
        val = data[row * ncols + idx];
    } else if (idx<static_cast<int>(ncols)-1) {
        val = data[row * ncols + idx]*(1-w) + data[row * ncols + idx + 1 ]*w;
    } else {
        val = std::numeric_limits<double>::quiet_NaN();
    }
    
    //std::cout << "row " << row << " " << col << " " << idx << " " << nrows << " " << ncols << " " << w << " " << val << std::endl;
    
    return val;
    
}

double nearest_interp_row( double * data, unsigned int nrows, unsigned int ncols, int row, double col ) {
    int idx = std::round( col );
    //std::cout << "row " << row << " " << col << " " << idx << " " << nrows << " " << ncols << std::endl;
    if (idx<0 || idx>=static_cast<int>(ncols)) { return std::numeric_limits<double>::quiet_NaN(); }
    return data[row * ncols + idx];
}

double linear_interp_col( double * data, unsigned int nrows, unsigned int ncols, double row, int col ) {
    
    int idx = std::floor( row );
    double w = row - idx;
    
    //std::cout << "col " << row << " " << idx << " " << col << " " << nrows << " " << ncols << " " << w << std::endl;
    if (row<-1 || row>=nrows) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (idx==-1) {
        return data[(idx+1) * ncols + col]*w;
    } else if (idx==static_cast<int>(nrows)-1) {
        return data[idx * ncols + col]*(1-w);
    } else if (std::abs(w) < std::numeric_limits<double>::epsilon()) {
        return data[idx * ncols + col];
    } else if (idx<static_cast<int>(nrows)-1) {
        return data[idx * ncols + col]*(1-w) + data[ (idx+1) * ncols + col]*w;
    } else {
        return std::numeric_limits<double>::quiet_NaN(); 
    }
}

double nearest_interp_col( double * data, unsigned int nrows, unsigned int ncols, double row, int col ) {
    int idx = std::round( row );
    //std::cout << "col " << row << " " << idx << " " << col << " " << nrows << " " << ncols << std::endl;
    if (idx<0 || idx>=static_cast<int>(nrows)) { return std::numeric_limits<double>::quiet_NaN(); }
    return data[idx * ncols + col];
}

double interp_row( Interpolation interp, double * data, unsigned int nrows, unsigned int ncols, int row, double col ) {
    if (interp==Interpolation::Nearest) {
        return nearest_interp_row( data, nrows, ncols, row, col );
    } else {
        return linear_interp_row( data, nrows, ncols, row, col );
    }
}

double interp_col( Interpolation interp, double * data, unsigned int nrows, unsigned int ncols, double row, int col ) {
    if (interp==Interpolation::Nearest) {
        return nearest_interp_col( data, nrows, ncols, row, col );
    } else {
        return linear_interp_col( data, nrows, ncols, row, col );
    }
}


Radon::Radon( double dx, double dy, Interpolation interpolation, Constraint constraint, IntegralMethod method, bool valid, bool intercept ) :
dx_(dx), dy_(dy), interpolation_(interpolation), constraint_(constraint),
integral_method_(method), valid_(valid), intercept_(intercept) {
    
    if (dx_<=0 || dy_<=0) {
        throw std::runtime_error("dx and dy need to be larger than 0.");
    }
    
}
    
Interpolation Radon::interpolation() const { return interpolation_; }
Constraint Radon::constraint() const { return constraint_; }
IntegralMethod Radon::integral_method() const { return integral_method_; }
bool Radon::valid() const { return valid_;}
bool Radon::intercept() const { return intercept_; }
double Radon::dx() const { return dx_; }
double Radon::dy() const { return dy_; }

void Radon::set_interpolation( Interpolation val ) { interpolation_ = val; }
void Radon::set_interpolation( std::string val ) { interpolation_ = interpolation_from_string( val ); }
void Radon::set_constraint( Constraint val ) { constraint_ = val; }
void Radon::set_constraint( std::string val ) { constraint_ = constraint_from_string( val ); }
void Radon::set_integral_method( IntegralMethod val ) { integral_method_ = val; }
void Radon::set_integral_method( std::string val ) { integral_method_ = integral_from_string( val ); }
void Radon::set_valid( bool val ) { valid_ = val; }
void Radon::set_intercept( bool val ) { intercept_ = val; }
void Radon::set_dx( double val ) { if (val<=0.) { throw std::runtime_error("dx needs to be larger than 0."); } else { dx_=val; } }
void Radon::set_dy( double val ) { if (val<=0.) { throw std::runtime_error("dy needs to be larger than 0."); } else { dy_=val; } }

void Radon::transform( double * data, unsigned int nrows, unsigned int ncols,
                    double * theta, unsigned int ntheta,
                    double * rho, unsigned int nrho,
                    double * result, uint16_t * n) const {
    
    double costheta, sintheta;
    double rhooffset, alpha, beta;
    int count;
    
    double xmin = -dx_*(nrows-1)/2.0; //x of lower left corner of matrix m (in local coordinates)
    double ymin = -dy_*(ncols-1)/2.0; //y of lower left corner of matrix m (in local coordinates)
    
    double L = std::sqrt(dx_*dx_+dy_*dy_);
    
    bool row_loop, valid;
    
    IndexPair range;
    double sum = 0, tmp = 0;
    
    //loop through all angles in theta
    for (unsigned int t=0; t<ntheta; ++t) {
        
        costheta = std::cos(theta[t]);
        sintheta = std::sin(theta[t]);
        
        row_loop = constraint_==Constraint::X || (constraint_==Constraint::No && ( std::abs(sintheta)>(dx_/ L ) ) ) ;
        
        rhooffset = xmin*costheta + ymin*sintheta;
        alpha = -(dx_/dy_)*(costheta/sintheta);
        
        if (!row_loop) { alpha = 1./alpha; }
    
        for (unsigned int r=0; r<nrho; ++r) {
            
            if (intercept_) { beta = rho[r]*costheta-rhooffset; } ///(dy_*sintheta);
            else { beta = rho[r]-rhooffset; } // /(dy_*sintheta);
            
            if (row_loop) { beta /= dy_*sintheta; }
            else { beta /= dx_*costheta; }
            
            if (row_loop) {
                range = indices( nrows, ncols, alpha, beta, interpolation_ );
                //range = IndexPair( { 0, nrows } );
            } else {
                range = indices( ncols, nrows, alpha, beta, interpolation_ );
                //range = IndexPair( { 0, ncols } );
            }
            
            count = std::max( 0, range.second-range.first );
            
            valid = false;
            n[0] = n[1] = 0;
            
            if ( count<=0 || ( valid_ && 
               ( (  row_loop && count<static_cast<int>(nrows) && std::abs(alpha*count)<(ncols-2) ) ||
                 ( !row_loop && count<static_cast<int>(ncols) && std::abs(alpha*count)<(nrows-2) ) ) ) ) {
                
                *result++ = std::numeric_limits<double>::quiet_NaN(); // NaN
                n+=2;
                // TODO: range[ ] = 0
                continue;
            }
            
            switch (integral_method_) {
                case IntegralMethod::Product :
                    sum = 1;
                    break;
                default :
                    sum = 0;
            }
            
            //std::cout << alpha << " " << beta << std::endl;
            //std::cout << range.first << " " << range.second << std::endl;
            
            for ( int q=range.first; q<range.second; ++q) {
                
                if (row_loop) {
                    tmp = interp_row( interpolation_, data, nrows, ncols, q, alpha * q + beta );
                } else {
                    tmp = interp_col( interpolation_, data, nrows, ncols, alpha * q + beta, q ) ;
                }
                
                if (std::isnan(tmp)) { continue; }
                
                //std::cout << tmp << std::endl;
                
                switch (integral_method_) {
                    case IntegralMethod::Integral :
                    case IntegralMethod::Sum :
                    case IntegralMethod::Mean :
                        sum += tmp;
                        break;
                    case IntegralMethod::LogSum :
                        sum += std::log(tmp);
                        break;
                    case IntegralMethod::Product :
                        sum *= tmp;
                        break;
                }
                
                if (valid) {
                    n[1] = q;
                } else {
                    valid = true;
                    n[0] = q;
                    n[1] = q;
                }
                
            }
            
            if (!valid) { sum = std::numeric_limits<double>::quiet_NaN(); }
            
            switch (integral_method_) {
                case IntegralMethod::Integral :
                    if (row_loop) {sum *= dx_ / std::abs( sintheta );}
                    else {sum *= dy_ / std::abs( costheta );}
                    break;
                case IntegralMethod::Mean :
                    sum /= count;
                    break;
                default :
                    break;
            }
            
            *result++ = sum;
            n += 2;
        }
            
    }
        
}


void Radon::slice( double * data,  unsigned int nrows, unsigned int ncols, 
                   double theta, double rho,
                   std::vector<double> & result, IndexPair & range ) const {
    
    double xmin = -dx_*(nrows-1)/2.0; //x of lower left corner of matrix m (in local coordinates)
    double ymin = -dy_*(ncols-1)/2.0; //y of lower left corner of matrix m (in local coordinates)
    
    double costheta = std::cos(theta);
    double sintheta = std::sin(theta);
    
    double rhooffset, alpha, beta;
    int count;
    
    double L = std::sqrt(dx_*dx_+dy_*dy_);
    
    result.clear();
    
    bool row_loop = constraint_==Constraint::X || (constraint_==Constraint::No && ( std::abs(sintheta)>(dx_/ L ) ) );
    
    rhooffset = xmin*costheta + ymin*sintheta;
    alpha = -(dx_/dy_)*(costheta/sintheta);
        
    if (!row_loop) { alpha = 1./alpha; }
    
    if (intercept_) { beta = rho*costheta-rhooffset; } ///(dy_*sintheta);
    else { beta = rho-rhooffset; } // /(dy_*sintheta);
    
    if (row_loop) { beta /= dy_*sintheta; }
    else { beta /= dx_*costheta; }
    
    if (row_loop) {
        range = indices( nrows, ncols, alpha, beta, interpolation_ );
    } else {
        range = indices( ncols, nrows, alpha, beta, interpolation_ );
    }
    
    count = std::max( 0, range.second-range.first );
    
    if ( count<=0 || ( valid_ && 
       ( (  row_loop && count<static_cast<int>(nrows) && std::abs(alpha*count)<(ncols-2) ) ||
         ( !row_loop && count<static_cast<int>(ncols) && std::abs(alpha*count)<(nrows-2) ) ) ) ) {
        
        range.first = 0;
        range.second = 0;
        return;
    }
    
    for ( int q=range.first; q<range.second; ++q) {
    
        if (row_loop) {
            result.push_back( interp_row( interpolation_, data, nrows, ncols, q, alpha * q + beta ) );
        } else {
            result.push_back( interp_col( interpolation_, data, nrows, ncols, alpha * q + beta, q ) );
        }
        
    }
    
}
    
    
