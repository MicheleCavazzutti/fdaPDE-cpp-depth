// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __DEPTH_H__
#define __DEPTH_H__

#include <fdaPDE/utils.h>
using fdapde::core::BlockFrame;
using fdapde::core::Triangulation;
using fdapde::core::Voronoi;
//using fdapde::core::BinaryMatrix; 

#include "../model_macros.h"
#include "../model_traits.h"
#include "../sampling_design.h"

#include <cmath>

// In this file, we provide the implementations needed to compute the partially observed integrated functional depth for multidimensional domains
// The file contains three calsses:
//	- Bivariate_Depth_Solver: computes the pointwise univariatebivariate  depths of set of functions with respect to a set of reference functions
//	- Depth_Solver: computes the pointwise univariate depths of set of functions with respect to a set of reference functions
//      - DEPTH<Triangulation>: provides all the routines for depth computation, as well as the computation of the functional boxplot. Relies on the Depth_solver class for the actual depth computation. Relyies on the Voronoi<Triangulation> class for Voronoi measures computation, if Voronoi integration is required.  

namespace fdapde {
  namespace models {
  
    // Depth_Solver class:
    // Class for the computation of univariate depths in the seeds/nodes of the triangulation. This class expects both the fit (referece) and pred functional data to be measured in the same set of points (i.e. the mesh nodesm that coincide with the Voronoi seeds); therefore, a seed based representation is required, which needs to be provided by external routines.
    // The class just stores the fit_data_m which represents the functional data used to estimate the pointwise distribution, and pred_data_, which are the functional data that need to be ranked. Not that, if pred_data_ = fit_data_, we obtain the depths of the functional data with respect to themselves, which is the typycal behaviour  expected in Dpeth computation routines. 
    // fit_mask_ and pred_mask_ store the NA patterns for the data, that may be missing in some noeds (partially observed functionaol data).
    // The class exposes the routines that allow to compute the univariate depths in each node. The univariate depth computation requires the ranking of the data in each point as a prestep, which is done in the private routine compute_rankings(). The rankings are stored internally for computational advantage.
    class Depth_Solver {
    private:
      const DMatrix<double> & fit_data_;  	// Reference functional data. Must have the same numbe rof columns of pred_data_. Size: n_train x n_nodes
      const DMatrix<bool> & fit_mask_;		// Reference data missing pattern (influences the univariate ranking in each node). Size: n_train x n_nodes
      DMatrix<double> pred_data_;		// Predictive functional data. Must have the same numbe rof columns of fit_data_. Size: n_pred x n_nodes
      DMatrix<bool> pred_mask_;			// Predictive data missing pattern (influences the univariate ranking in each node). Size: n_pred x n_nodes
      int n_train;				
      int n_pred;
      int n_nodes;

      DMatrix<int> rankings;			// Matrix that stores the ranking of the predictive data with respect to the fit ones (the ranks may be repeated). Size n_pred x n_nodes
      DVector<int> NA_number;			// Number of missing train data in each node. Size: n_nodes

      bool are_rankings_computed = false; // Boolean flag indicating if rakings have been computed already or if computation is needed

      // Computes the relative ranks of the predictive data with respect to the train data, and stores them in the rankings matrix
      void compute_rankings(){
	rankings.resize(n_pred, n_nodes);
	NA_number.resize(n_nodes);
	
	// Initialization
	for(auto i =0; i<n_nodes; i++){
	  NA_number(i)=0;
	}

        for(auto j =0; j<n_nodes; j++){
	  for(auto i =0; i<n_pred; i++){
	    if(pred_mask_(i,j)==true){ // Predict datum is NA
	      rankings(i,j)=n_train;
	    }else{
	      int count_down=0;
	      for(auto k =0; k<n_train; k++){
		if(fit_mask_(k,j)==false && fit_data_(k,j) < pred_data_(i,j)){ // Fit datum is not NA and is lower than the predict datum
		  count_down++;
		}else{
		}
		rankings(i,j)=count_down;
	      }
	    }
	  }
	  for(auto k=0; k< n_train; k++){ // Count the NA in train data
	    if(fit_mask_(k,j)==true){
	      this->NA_number(j)++;
	    }
	  } 
	}
	
	are_rankings_computed=true;
	
	return;
      }

    public:
      Depth_Solver(const DMatrix<double> & fit_data, const DMatrix<bool> & fit_mask): fit_data_(fit_data), fit_mask_(fit_mask){ // Constructor
	n_train = fit_data_.rows();
	n_nodes = fit_data_.cols();	
      }
      
      // Setters
      void set_pred_data(const DMatrix<double> & pred_data){pred_data_ = pred_data;
	n_pred = pred_data.rows();
	are_rankings_computed = false; // NB we need to reset the flag, because we have new preditive data
      }
      void set_pred_mask(const DMatrix<bool> & pred_mask){pred_mask_ = pred_mask;}

      // Routine that computes the univariate simplicial depth in the j-th node of the triangulation (Voronoi seed). 
      // Requires the ranking of the predictive data which is carried out in the routine compute_rankings and stored in the rankings matrix
      DVector<double> compute_SD(int j){
	if(!are_rankings_computed){
	  this->compute_rankings(); // Compute the rankings of the data to be evaluated ( that may involve both the single fit or the single pred) data w.r.t. the fit data already encoded.
	}
        
	DVector<double> result;
	result.resize(n_pred);
      
	for(auto i = 0; i< n_pred; i++){
	  if(this->n_train - this->NA_number(j) - this->rankings(i,j) > 0 ){ // Check, lui non sicuro
	    result(i) = (double) 2*(this->n_train - this->NA_number(j) - this->rankings(i,j) - 1)*(this->rankings(i,j) + 1)/(double) ((this->n_train - this->NA_number(j) )*(this->n_train - this->NA_number(j) ));
	  }
	  else{
	    result(i) = 0;
	  }
	}
      
	return result;
      }
      
      // Routine that computes the univariate haflspace depth in the j-th node of the triangulation (Voronoi seed). 
      // Also computes the corresponding Hypograph and Epigraph univariate depths, which employed in the MHRD computation. 
      // Requires the ranking of the predictive data which is carried out in the routine compute_rankings and stored in the rankings matrix
      DMatrix<double> compute_HD(int j){ 
	if(!are_rankings_computed){
	  this->compute_rankings(); // Compute the rankings of the data to be evaluated ( that may involve both the single fit or the single pred) data w.r.t. the fit data already encoded.
	}
      
	DMatrix<double> result;
	result.resize(n_pred,3);
	
	//initialization
	for(auto i = 0; i< n_pred; i++){
	  result(i,0) = 0;
	  result(i,1) = 0;
	  result(i,2) = 0;
	}
      
	for(auto i = 0; i< n_pred; i++){
	  if(this->n_train - this->NA_number(j) - this->rankings(i,j) > 0 ){ // Check, lui non sicuro
	    result(i,1) = (double) (this->n_train - this->NA_number(j) - rankings(i,j) - 1 )/(double) (this->n_train - this->NA_number(j)); // Epigraph
	    result(i,2) = (double) (this->rankings(i,j))/(double) (this->n_train - this->NA_number(j)) ; // Hipograph
	    result(i,0) = std::min(result(i,1), result(i,2)); // MHRD local (discarded, but may be useful afterwards)
	  }
	  else{
	    result(i,0) = 0;
	    result(i,1) = 0;
	    result(i,2) = 0;
	  }
	}
      
	return result;
      }
    
      // Routine that computes the univariate Tukey depth in the j-th node of the triangulation (Voronoi seed). 
      // Requires the ranking of the predictive data which is carried out in the routine compute_rankings and stored in the rankings matrix
      DVector<double> compute_FMD(int j){
	if(!are_rankings_computed){
	  this->compute_rankings(); // Compute the rankings of the data to be evaluated ( that may involve both the single fit or the single pred) data w.r.t. the fit data already encoded.
	}
              
	DVector<double> result;
	result.resize(n_pred);
	
	for(auto i = 0; i< n_pred; i++){
	  if(this->n_train - this->NA_number(j) - this->rankings(i,j) > 0 ){ // Check, lui non sicuro
	    result(i) = 1 - (double) (2*(this->rankings(i,j) + 1)-1)/(double) (2*(this->n_train - this->NA_number(j)));
	  }
	  else{
	    result(i) = 0;
	  }
	}  
	return result;
      }
    };

    // Bivariate_Depth_Solver class:
    // Class for the computation of bivariate depths.
    // The class just stores the fit_data_ which represents the bivariate data used to estimate the pairwise distribution, and pred_data_, which are the functional data that need to be ranked. Not that, if pred_data_ = fit_data_, we obtain the depths of the bivariate data with respect to themselves, which is the typycal behaviour  expected in Depth computation routines. 
    class Bivariate_Depth_Solver {
    private:
      DMatrix<double> fit_data_;         	// Reference bivariate data. Size: n_fit_points x 2.
      DMatrix<bool> fit_mask_;           	// Reference bivariate data mask. Size: n_fit_points x 2.
      DMatrix<double> pred_data_;		// Predictive bivariate data. Size: n_pred_points x 2.
      DMatrix<bool> pred_mask_;	        	// Predictive bivariate data mask. Size: n_pred_points x 2.
      int n_fit_points;				
      int n_pred_points;
      std::vector<double> alpha;                // Angles that are used in Simplicial depth computation, will be set multiple times
      std::vector<int> valid_fit_indices;       // set ofthe valid indices, will be set multiple times

    public:
      Bivariate_Depth_Solver() = default; // default constructor
      
      // Setters
      void set_fit_data(const DMatrix<double> & fit_data, const DMatrix<bool> & fit_mask){
	fit_data_ = fit_data;
	fit_mask_ = fit_mask;
	n_fit_points = fit_data.rows();
	alpha.reserve(n_fit_points); // Preallocate alpha for simplicial computation
	valid_fit_indices.reserve(n_fit_points);  // Preallocate valid_fit_indices for simplicial computation
      }
      
      void set_pred_data(const DMatrix<double> & pred_data, const DMatrix<bool> & pred_mask){
	pred_data_ = pred_data;
	pred_mask_ = pred_mask;
	n_pred_points = pred_data.rows();
      }

      // Routine that computes the bivariate simplicial depth for a set of points.
      DVector<double> compute_SD() {
	DVector<double> depths(n_pred_points);
	const double eps = 1e-12; // Numerical tolerance
	const double PI = std::acos(-1.0);

	valid_fit_indices.clear();
	for (int i = 0; i < n_fit_points; ++i) {
	  // Keep only points with no missing values in both coordinates
	  if (!fit_mask_(i, 0) && !fit_mask_(i, 1)) {
            valid_fit_indices.push_back(i);
	  }
	}

	long long n_valid_fit = valid_fit_indices.size();
	// Denominator: total number of 3-point combinations (n choose 3)
	const double total_simplices =
	  static_cast<double>(n_valid_fit) *
	  (n_valid_fit - 1) *
	  (n_valid_fit - 2) / 6.0;

	for (int zi = 0; zi < n_pred_points; ++zi) {
	  if (pred_mask_(zi, 0) || pred_mask_(zi, 1)) {
            depths(zi) = -1.0;
            continue;
	  }

	  if (total_simplices < 1.0) {
            depths(zi) = 0.0;
            continue;
	  }

	  const double zx = pred_data_(zi, 0);
	  const double zy = pred_data_(zi, 1);

	  alpha.clear();
	  long long nt = 0;

	  for (int idx : valid_fit_indices) {
            double dx = fit_data_(idx, 0) - zx;
            double dy = fit_data_(idx, 1) - zy;
            double d2 = dx * dx + dy * dy;

            // Count points coinciding with z
            if (d2 <= eps * eps) {
	      nt++;
            } else {
	      // Store angular direction of non-coinciding points
	      alpha.push_back(std::atan2(dy, dx));
            }
	  }

	  const long long nn = alpha.size();

	  if (nn < 2) {
            // If there are not enough non-coinciding points,
            // the depth depends only on coinciding points
            if (nt + nn >= 3) {
	      // Simplified combinatorial count when most points coincide with z
	      unsigned long long total_inc =
		(unsigned long long)nt * (nt - 1) * (nt - 2) / 6 +
		(unsigned long long)nt * (nt - 1) / 2 * nn;
	      depths(zi) = static_cast<double>(total_inc) / total_simplices;
            } else {
	      depths(zi) = 0.0;
            }
            continue;
	  }

	  std::sort(alpha.begin(), alpha.end());

	  // Duplicate angles to handle circularity without conditional branching
	  for (long long i = 0; i < nn; ++i) {
            alpha.push_back(alpha[i] + 2.0 * PI);
	  }

	  unsigned long long outside = 0;
	  long long j = 0;

	  // Sliding window (two-pointers) algorithm
	  for (long long i = 0; i < nn; ++i) {
            if (j <= i) j = i + 1;
            // Find the farthest point within a semicircle (PI radians)
            while (j < 2 * nn && (alpha[j] - alpha[i]) < PI - eps) {
	      j++;
            }
            long long m = j - i - 1;
            if (m >= 2) {
	      outside += (unsigned long long)m * (m - 1) / 2;
            }
	  }

	  // Triangles containing the origin (using only non-coinciding points)
	  unsigned long long inside_no_nt = 0;
	  unsigned long long combinations_nn_3 =
            (unsigned long long)nn * (nn - 1) * (nn - 2) / 6;
	  if (combinations_nn_3 > outside) {
            inside_no_nt = combinations_nn_3 - outside;
	  }

	  // Add contributions from coinciding points (nt)
	  // 1. Triangles with 1 coinciding point and 2 non-coinciding points: C(nt,1) * C(nn,2)
	  // 2. Triangles with 2 coinciding points and 1 non-coinciding point: C(nt,2) * C(nn,1)
	  // 3. Triangles with 3 coinciding points: C(nt,3)
	  unsigned long long inside_with_nt =
            inside_no_nt +
            (unsigned long long)nt * (nn * (nn - 1) / 2) +
            ((unsigned long long)nt * (nt - 1) / 2) * nn +
            (unsigned long long)nt * (nt - 1) * (nt - 2) / 6;

	  depths(zi) = static_cast<double>(inside_with_nt) / total_simplices;
	}

	return depths;
      }
      
    };

    // depth model
    // This template class is the baseline model for depth computation. The template parameter D represents the Triangulation of the support used to represent the geometry of the problem, that is sotred in the Domain variable. The class stores the reference functional data used to compute the distribution of the functional depth (fit data). The functional data may be only partially observable (that is, missing in some locations), a feature that may be expressed using the NA masks associated to the data. Moreover, the measurement locations may vary for each statistical unit to another. The main utility is the method solve, that computes the partially observed integrated functional depth for the fit data with respect to themselves. Thhe method allows for two types of integral approximations: Voronoi based approximation and FEM based approximation. Through the method predict() is also possible to compute the depths of some novel functional data with respect to the fit data, possibly carachterized by different locations and different missing patterns. The class also carries out the computations of the objects needed for the construction of a functional boxplot based on fit data.
    template <typename D> 							// Domain type
    class DEPTH {
    public:
      using SpaceDomainType = D;          					// triangulated spatial domain
      using VoronoiTessellation = fdapde::core::Voronoi<SpaceDomainType>;    	// Voronoi tessellation of the spatial domain (available only if Voronoi integrations is required)

      DEPTH() = default; // Check
      // constructor that takes as imput the triangulation of the support (see the file triangulation.h to understand how to build one)
      DEPTH(const D & domain):domain_(domain){};

      // setters
      void set_locations(const std::vector<DMatrix<double>> &  locations){  locations_ =  locations ; }
      void set_pred_locations(const std::vector<DMatrix<double>> &  locations_pred){  locations_pred_ =  locations_pred ; }
      void set_roi(const DVector<int> & roi ) { roi_ = roi; }
      void set_depth_types(const DVector<int> & depth_types ) { depth_types_ = depth_types; }
      void set_pred_depth_types(const DVector<int> & depth_types ) { pred_depth_types_ = depth_types; }
      void set_train_functions(const std::vector<DVector<double>> &  train_functions){  train_functions_ =  train_functions ; }
      void set_train_matrix_NA(const std::vector<DVector<bool>> &  NA_matrix){  train_matrix_NA_ =  NA_matrix ; } 
      void set_pred_functions(const std::vector<DVector<double>> & pred_functions) { pred_functions_ = pred_functions ; }
      void set_pred_matrix_NA(const std::vector<DVector<bool>> &  NA_matrix_pred){  pred_matrix_NA_ =  NA_matrix_pred ; } 
      void set_phi_function_evaluation(const DVector<double> & phi_function_evaluation ) { phi_function_evaluation_ = phi_function_evaluation;} 
      void set_external_voronoi_measures(const DVector<double> & external_voronoi_measures ) { external_voronoi_measures_ = external_voronoi_measures;} 
      void set_int_method(int int_method ) { int_method_ = int_method;} 
      
      void set_seed_based_r_fit(const DMatrix<double> &  seed_based_r_fit ) { return seed_based_r_fit_ =  seed_based_r_fit ; }
      void set_seed_based_r_pred(const DMatrix<double> &  seed_based_r_pred ) { return seed_based_r_pred_ =  seed_based_r_pred; } 
      
      void set_IFD_fit(const DMatrix<double> & IFD_fit ) { IFD_fit_ = IFD_fit; }
      void set_IFD_pred(const DMatrix<double> & IFD_pred ) { IFD_pred_ = IFD_pred; }

      // getters
      const SpaceDomainType & domain() const { return domain_; }                                   
      const VoronoiTessellation & voronoi() const { return voronoi_; }                             
      
      const DVector<int> & depth_types() const { return depth_types_; }
      const DVector<int> & pred_depth_types() const { return pred_depth_types_; }
      const std::vector<DMatrix<double>> & locations() const { return locations_; }
      const std::vector<DVector<double>> & train_functions() const { return train_functions_; }
      const std::vector<DVector<bool>> & train_pattern_NA() const { return train_matrix_NA_; }
      const std::vector<DVector<double>> & pred_functions() const { return pred_functions_; }
      const std::vector<DVector<bool>> & pred_pattern_NA() const { return pred_matrix_NA_; }
      const DVector<double> & phi_function_evaluation() const { return phi_function_evaluation_; } // Returns phi function used to evaluate the IFD phi in the nodes of the functions
      const DVector<double> & external_voronoi_measures() const { return external_voronoi_measures_; } // return the voronoi measures, only for 2.5D and 3D domains
      int int_method() const { return int_method_; }
      
      const DVector<double> & density_vector(){return observation_density_vector_; }
      const DMatrix<double> & seed_based_r_fit() const { return seed_based_r_fit_; }  
      const DMatrix<bool> & seed_based_r_fit_NA() const { return seed_based_r_fit_NA_; } 
      const DMatrix<double> & seed_based_r_pred() const { return seed_based_r_pred_; }
      const DMatrix<bool> & seed_based_r_pred_NA() const { return seed_based_r_pred_NA_; } 
      
      const DMatrix<double> & IFD_fit() const { return IFD_fit_; }
      const DMatrix<double> & IFD_pred() const { return IFD_pred_; } 
      const DVector<double> & mepi_fit() const { return mepi_fit_; }
      const DVector<double> & mhypo_fit() const { return mhypo_fit_; }
      const DVector<double> & mepi_pred() const { return mepi_pred_; }
      const DVector<double> & mhypo_pred() const { return mhypo_pred_; }
      
      const DMatrix<double> & medians() const { return medians_;} 			
      const DMatrix<bool> & medians_NA() const { return medians_NA_; } 		
      const DMatrix<double> & first_quartile() const { return first_quartile_; }
      const DMatrix<bool> & first_quartile_NA() const { return first_quartile_NA_; }
      const DMatrix<double> & third_quartile() const { return third_quartile_; } 
      const DMatrix<bool> & third_quartile_NA() const { return third_quartile_NA_; } 		        
      const DMatrix<double> & up_whisker() const { return up_whisker_; }
      const DMatrix<bool> & up_whisker_NA() const { return up_whisker_NA_; }	        
      const DMatrix<double> & low_whisker() const { return low_whisker_; } 
      const DMatrix<bool> & low_whisker_NA() const { return low_whisker_NA_; } 		        
      const DMatrix<bool> & outliers() const { return outliers_; }                       
      
      void init() { // Initialization routine, prepares the environment for the solution of the problem.
	
	if(int_method_ == -1){ // Voronoi-based integration, we need to access to the Voronoi representation of the domain
	  VoronoiTessellation voronoi(domain_); // Compute voroni tessellation of the model (for the moment only <2,2>, <1,1> meshes are available)
	  this->voronoi_ = voronoi;  // Store the object in the internal memory for future use
	}
	
	// At first compute the seed-based representation of data; this needs to be done after the voronoi has been computed if int_method_ == 0.
	this->compute_seed_based_representation_fit();

	// If depth_types_ inclused PDI- depths, compute the node patches for FEM-0 or the node rings for Voronoi
	this->compute_seed_patches();

	// initialize the depth structures
	int n_train = this->seed_based_r_fit_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	// Now we have available in seed_based_r_fit and seed_based_r_fit_NA_ the computed seed-based (Voronoi or FEM) representation of the matrix.
	// We can compute the empirical distribution (Q(p)) in the voronoi nodes, using the NA pattern. We provide equal weight to each element
	observation_density_vector_.resize(domain_.n_nodes());
	for (auto i=0; i<domain_.n_nodes(); i++){// for each node of the mesh, count how many times a cell has been observed in the Voronoi mask. 
	  auto obs_element = seed_based_r_fit_NA_.col(i);
	  observation_density_vector_(i) = n_train - obs_element.count();
	}
	observation_density_vector_ = observation_density_vector_ / n_train;

	this->IFD_fit_.resize(n_train, this->depth_types_.size());
	this->mepi_fit_.resize(n_train);
	this->mhypo_fit_.resize(n_train);

	// initialization
	for (auto i =0; i < n_train; i++){
	  mepi_fit_(i) = 0;
	  mhypo_fit_(i) = 0;
	  for(auto j =0 ; j<this->depth_types_.size(); j++){
	    IFD_fit_(i,j)=0;
	  }
 	}

	// Ready to solve the problem
	return; 
      } 
      
      void solve() { //  Compute the integrated depths and the outputs that will be returned (save outputs in a df), fill output 
	
	bool single_integral_present = false;
	bool double_integral_present = false;
      
	for(auto j = 0 ; j < this->depth_types_.size(); j++){
	  if(this->depth_types_(j) == 1 || this->depth_types_(j) == 2 || this->depth_types_(j) == 3){
	    single_integral_present = true;
	  }
	  if(this->depth_types_(j) == 4 || this->depth_types_(j) == 5){ // DI-Depth, PDI-Depth
	    double_integral_present = true;
	  }
	}

	// Call the solvers for the single or double integral cases
	if(single_integral_present){
	  this->solve_single_integral_case();
	}
	if(double_integral_present){
	  this->solve_double_integral_case();
	}
      
	// only for fit functions, one can also compute the functional boxplot quantities. In principle, this may also be done in R, but here is faster
	this->compute_functional_boxplot();

	return; 
      }

      void solve_single_integral_case() { //  Compute the integrated depths and the outputs that will be returned (save outputs in a df), fill output 
      
	int n_train = this->seed_based_r_fit_.rows();
	int n_nodes = this->domain_.n_nodes(); 
	
	Depth_Solver solver(this->seed_based_r_fit_, this->seed_based_r_fit_NA_); // This solver uses the Voronoi representations of the fit functions to estimate the empirical measure.
      
	solver.set_pred_data(this->seed_based_r_fit_);
	solver.set_pred_mask(this->seed_based_r_fit_NA_);
      
	DMatrix<double> point_depth;
	DMatrix<double> point_aux;
	
	point_depth.resize(n_train, this->depth_types_.size()); // this will contain the point depth, for each element 
	point_aux.resize(n_train, 2); // this contains the computed point auxiliary indices, such as MEPI or MHYPO
	
	// initialization
	for (auto i =0; i < n_train; i++){
	  point_aux(i,0) = 0;
	  point_aux(i,1) = 0;
	  for(auto j =0 ; j<this->depth_types_.size(); j++){
	    point_depth(i,j)=0; // Note: if j refers to a type of depth with double integral, we will leave the point depth tozero, so not to affect the output of the double integral solver
	  }
 	}
	
	// weighting function denominator
	DVector<double> weight_den;
	
	// initialization 
	weight_den.resize(n_train);
	for(auto i = 0; i< n_train; i++){
	  weight_den(i)=0;
	}
	
	if(this->int_method_ == -1){ // Voronoi case
      
	  for (auto i=0; i<n_nodes; i++){
	    // extract the measure of the Voronoi cell 
	    double measure = this->voronoi_.cell(i).measure();
	    if((this->voronoi_.local_dim == 2 && this->voronoi_.embed_dim == 3) || (this->voronoi_.local_dim == 3 && this->voronoi_.embed_dim == 3)){ // In the 2.5D and 3D case we resort t external Voronoi measures
	      measure  = this->external_voronoi_measures_[i];
	    }

	    for(auto k =0; k<n_train; k++){
	      if(!seed_based_r_fit_NA_(k,i)){
		weight_den(k) = weight_den(k) + measure * this->phi_function_evaluation_(i);
	      }
	    }
      
	    for (auto j=0; j<this->depth_types_.size(); j++){
	      switch(depth_types_(j)) { // Basing on the type of depth required
	      case 1: //SD: simplicial univariate depth
		{
		  point_depth.col(j) = solver.compute_SD(i) * this->phi_function_evaluation_(i);
	      
		}
		break;
	      
	      case 2: // FMD
		{
		  point_depth.col(j) = solver.compute_FMD(i);
		}  
		break;
	    
	      case 3: // MHRD
		{
		  DMatrix<double> MHRD_solution = solver.compute_HD(i) * this->phi_function_evaluation_(i); // Note: MHRD is defined as the global minimum between the MEPI and MHIPO. So it will be overwritten afterwards.
		  point_depth.col(j) = MHRD_solution.col(0);
		  point_aux = MHRD_solution.rightCols(2); // Epigraph and hypograph indices
            
		  mepi_fit_ = mepi_fit_ + point_aux.col(0)*measure;
		  mhypo_fit_ = mhypo_fit_ + point_aux.col(1)*measure;
		}  
		break;
	    
	      default:
		{} // This case also encapsulates the double integral depths, where essentially we do not want to modify the depth computed in the double integral solver
		break;
	     
	      }
      
      
	    }
	  
	    IFD_fit_ = IFD_fit_ + point_depth*measure;
      
	  }
	}else{ // FEM P0 case
	
	  // Build storage matrices needed for the FEM P0 computation
	  // # nodes x depth_types.size();
	  DMatrix<double> depths_storage;
	  depths_storage.resize(n_train, n_nodes * depth_types_.size()); // One column for each node and each type of depth required
	  depths_storage.setZero();
	  DMatrix<double> mepi_storage;
	  mepi_storage.resize(n_train, n_nodes);
	  mepi_storage.setZero();
	  DMatrix<double> mhypo_storage;
	  mhypo_storage.resize(n_train, n_nodes);
	  mepi_storage.setZero();
	
	  // Extract the depths from the univariate depth solver before FEM computation.
	  for (auto i=0; i<n_nodes; i++){
	    for (auto j=0; j<this->depth_types_.size(); j++){
	      switch(depth_types_(j)) { // Basing on the type of depth required
	      case 1: //SD: simplicial univariate depth
		{
		  depths_storage.col(i+j*n_nodes) = solver.compute_SD(i) * this->phi_function_evaluation_(i);
	      
		}
		break;
	      
	      case 2: // FMD
		{
		  depths_storage.col(i+j*n_nodes) = solver.compute_FMD(i);
		}  
		break;
	    
	      case 3: // MHRD
		{
		  DMatrix<double> MHRD_solution = solver.compute_HD(i) * this->phi_function_evaluation_(i); 
		  depths_storage.col(i+j*n_nodes) = MHRD_solution.col(0);
		  point_aux = MHRD_solution.rightCols(2); // Epigraph and Hypograph indices
            
		  mepi_storage.col(i) = point_aux.col(0);
		  mhypo_storage.col(i) = point_aux.col(1);
		}  
		break;
	    
	      default: // this case encapsulates the situation when part of the depths required are double integral depths
		{
		}
		break;
	     
	      }
      
	    }
	  }
	
	  // Aux variables
	  DMatrix<double> barycenters_depth;
	  barycenters_depth.resize(n_train,this->depth_types_.size() + 2); // Depths + mepi + mhypo
	  double barycenters_weights = 0;
	  DVector<bool> missing_cell;
	  missing_cell.resize(n_train);
	
	  // Perform FEM computation for both the numerator and denominator of FEMD
	  for(typename D::cell_iterator iter = domain_.cells_begin(); iter != domain_.cells_end(); ++iter){ // For each element (triangle or thetahedron in the Triangulation)
	    // initialize
	    barycenters_depth.setZero();
	    barycenters_weights = 0;
	    missing_cell.setConstant(false);
	
	    // Get the measure of the simplex
	    double measure = iter->measure();
	  
	    // Extract the nodes indices
	    DVector<int> node_ids = iter->node_ids();
	  
	    // Compute the barycenters
	    for(int node_idx = 0; node_idx < node_ids.size(); node_idx++){
	      for(int j = 0; j < this->depth_types_.size(); j++){
		barycenters_depth.col(j) = barycenters_depth.col(j) + depths_storage.col(node_ids(node_idx) +j*n_nodes);
	      }
	      barycenters_depth.col(this->depth_types_.size()) = barycenters_depth.col(this->depth_types_.size()) + mepi_storage.col(node_ids(node_idx));
	      barycenters_depth.col(this->depth_types_.size()+1) = barycenters_depth.col(this->depth_types_.size()+1) + mhypo_storage.col(node_ids(node_idx));
	      barycenters_weights = barycenters_weights + this->phi_function_evaluation_(node_ids(node_idx));
	      for(auto k=0; k < n_train; k++){
		if(seed_based_r_fit_NA_(k,node_ids(node_idx)) == true){
		  missing_cell(k) = true;
		}
	      }
	    }
	    barycenters_depth = barycenters_depth / node_ids.size();
	    barycenters_weights = barycenters_weights / node_ids.size();
	  
	    // Add the elements to the overall integral
	    for(auto k=0; k < n_train; k++){
	      if(missing_cell(k)==false){ // If the function is missing in the cell k, just skip it (non-exisiting in the integral)
		weight_den(k) = weight_den(k) + barycenters_weights * measure;
	  
		for(int j =0; j < this->depth_types_.size(); j++){
		  IFD_fit_(k,j) = IFD_fit_(k,j) + barycenters_depth(k,j)*measure;
		}
		mepi_fit_(k) = mepi_fit_(k) + barycenters_depth.col(this->depth_types_.size())(k)*measure;
		mhypo_fit_(k) = mhypo_fit_(k) + barycenters_depth.col(this->depth_types_.size()+1)(k)*measure;
	      }
	    }
	  }
	
	}
	
	
	for(auto j=0; j < this->depth_types_.size();j++){ 
	  if(depth_types_(j)==3){// 3==MHRD The minimum between epigraph and hipograph indices
	    // fill the MHRD column with the minimum between between MEPI and MIPO. Note: check wether min is before or after integral!!!
	    for(auto k=0; k< n_train; k++){ // for every functional datum
	      mepi_fit_(k) = mepi_fit_(k) / weight_den(k);
	      mhypo_fit_(k) = mhypo_fit_(k) / weight_den(k);
	      IFD_fit_(k,j) = std::min(mepi_fit_(k), mhypo_fit_(k));  // Check that the std::min are appropriate in vector!!
	    }
	  }else{
	    if(depth_types_(j)==1){ // 1==SD: simplicial univariate depth
	      for(auto k=0; k< n_train; k++){ // for every functional datum
		IFD_fit_(k,j) = IFD_fit_(k,j) / weight_den(k);
	      }
	    }
	  }
	}

	return; 
      }

      void solve_double_integral_case(){
	int n_train = this->seed_based_r_fit_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	Bivariate_Depth_Solver solver; // This solver will compute the multivariate depth for each location couple; Will be recycled for each couple, so to avoid multiple dynamic memory allocation
	DMatrix<double> aux_bivariate_data_fit;
	DMatrix<bool> aux_bivariate_data_mask;
	DMatrix<double> aux_couple;
	DMatrix<bool> aux_couple_mask;
	aux_bivariate_data_fit.resize(n_train,2);
	aux_bivariate_data_mask.resize(n_train,2);
	aux_couple.resize(1,2);
	aux_couple_mask.resize(1,2);

	// weighting function denominator
	DVector<double> weight_den; // We will need to fill this while performing the integrals
	
	// Auxiliary variables to store the bivariate depths
	DMatrix<float> depths_storage;
	depths_storage.resize(n_nodes, n_nodes); // One column for each node and each type of depth required // To be initialized in each cycle for functional datum in fit
	
	// Cycle over the required depth types; here the logic differs from the one of the single integral case
	// Compute the depth for each double integral depth type and store it in IFD_fit_ 
	for(auto j = 0; j < this->depth_types_.size(); j++){
	  if(this->depth_types_(j)==1 || this->depth_types_(j)==2|| this->depth_types_(j)==3){ // It is not a double integral, skip it
	    continue; 
	  }
	  
	  // initialization of weight den
	  weight_den.resize(n_train);
	  for(auto k = 0; k < n_train; k++){
	    weight_den(k)=0;
	  }
	  
	  // Matrix that stores the result of the second integral over the domain for each node, and for each train function
	  DMatrix<double> expectations_at_nodes;
	  expectations_at_nodes.resize(n_train, n_nodes);
	  expectations_at_nodes.setConstant(0);
	  
	  // Matrix that stores the result of the second integral for the weight function over the domain for each node, and for each train function
	  DMatrix<double> expectations_weight;
	  expectations_weight.resize(n_train, n_nodes);
	  expectations_weight.setConstant(0);
	  
	  // for each functional datum, store the multivariate depths; then compute the second integral and store the result for each node into Expectation_at_nodes
	  for(auto k = 0; k < n_train; k++){
	    // initialize depth_storage to -1 (unfeasible value) in order to understand if we have computed a bivariate depth already
	    depths_storage.setConstant(0);

	    // Compute and store the bivariate depths needed
	    for(auto node_1 = 0; node_1 < n_nodes; node_1++){
	      for(auto node_2 = 0; node_2 <= node_1; node_2++){
		if(this->depth_types_(j)==4 ||( this->depth_types_(j)==5 && seed_rings_[node_1].contains(node_2))){ // Either we are computing DI-SD and we need to compute all the couples, or PDI-SS, and we restrict to the node ring or ROI, for each node_1
		  if(!seed_based_r_fit_NA_(k,node_1) && !seed_based_r_fit_NA_(k,node_2)){ // Both the functional evaluations are present, we can compute the multivariate depth with respect to the nonmissing data
		    // set the couple of which we need to compute the multivariate depth
		    aux_couple(0,0) = seed_based_r_fit_(k,node_1);
		    aux_couple(0,1) = seed_based_r_fit_(k,node_2);
		    aux_couple_mask(0,0) = seed_based_r_fit_NA_(k,node_1);
		    aux_couple_mask(0,1) = seed_based_r_fit_NA_(k,node_2);

		    // Set the corresponding bivariate fit data with respect to which we compute the multivariate depth
		    aux_bivariate_data_fit.col(0) = seed_based_r_fit_.col(node_1);
		    aux_bivariate_data_fit.col(1) = seed_based_r_fit_.col(node_2);
		    aux_bivariate_data_mask.col(0) = seed_based_r_fit_NA_.col(node_1);
		    aux_bivariate_data_mask.col(1) = seed_based_r_fit_NA_.col(node_2);
		  
		    // Set the mutlivariate depth solver
		    solver.set_fit_data(aux_bivariate_data_fit, aux_bivariate_data_mask);
		    solver.set_pred_data(aux_couple, aux_couple_mask);

		    // Compute and store the multivariate depth using the bivariate solver
		    depths_storage(node_1, node_2) = solver.compute_SD()(0) * this->phi_function_evaluation_(node_1) * this->phi_function_evaluation_(node_2) ;
		    depths_storage(node_2, node_1) = depths_storage(node_1, node_2); // The multivariate depth is symmetric
		  }// Otherwise just skip the couple, will allso be skipped in the integral
		}
		
	      }
	    }

	    // Second integral computation
	    if(this->int_method_ == -1){ // Voronoi case
	      // Perform Voronoi computation for both the numerator and denominator of the DI depth
	      for(auto node_1 = 0; node_1 < n_nodes; node_1++ ){ // For each node of the triangulation (and seed of the dual Voronoi tessellation)
		for(auto node_2 = 0; node_2 < n_nodes; node_2++ ){ // For each node of the triangulation (and seed of the dual Voronoi tessellation)
		  if(this->depth_types_(j)==4 ||( this->depth_types_(j)==5 && seed_rings_[node_1].contains(node_2))){ // Either we are computing DI-SD and we need to compute all the couples, or PDI-SS, and we restrict to the node ring or ROI, for each node_1
		    // Add the elements to the expectations for each node
		    if(!seed_based_r_fit_NA_(k,node_1) && !seed_based_r_fit_NA_(k,node_2)){// only if all the nodes in the simplex were present, otherwise skip the cell in this integral
		      // extract the measure of the Voronoi cell 
		      double measure = this->voronoi_.cell(node_2).measure();
		      if((this->voronoi_.local_dim == 2 && this->voronoi_.embed_dim == 3) || (this->voronoi_.local_dim == 3 && this->voronoi_.embed_dim == 3)){ // In the 2.5D and 3D case we resort t external Voronoi measures
			measure  = this->external_voronoi_measures_[node_2];
		      }
		      expectations_at_nodes(k,node_1) = expectations_at_nodes(k,node_1) + depths_storage(node_1,node_2)*measure;
		      expectations_weight(k,node_1) = expectations_weight(k,node_1) + this->phi_function_evaluation_(node_1) * this->phi_function_evaluation_(node_2) * measure;
		    }
		  }
		}
	      }
	    }else{ // FEM-0 case
	      // Aux variables that sum up in the expectations
	      DVector<double> barycenters_depth;
	      barycenters_depth.resize(n_nodes); 
	      DVector<double> barycenters_weights;
	      barycenters_weights.resize(n_nodes);
	      bool missing_cell = false;
	      int cell_index = 0;
	
	      // Perform FEM computation for both the numerator and denominator of the DI depth
	      for(typename D::cell_iterator iter = domain_.cells_begin(); iter != domain_.cells_end(); ++iter, ++cell_index){ // For each element (triangle or thetahedron in the Triangulation)
		// initialize
		barycenters_depth.setZero();
		barycenters_weights.setZero();
		missing_cell=false;
	
		// Get the measure of the simplex
		double measure = iter->measure();
	  
		// Extract the nodes indices
		DVector<int> node_ids = iter->node_ids();
	  
		// Compute the barycenters
		for(int node_idx = 0; node_idx < node_ids.size(); node_idx++){
		  barycenters_depth = barycenters_depth + (depths_storage.col(node_ids(node_idx))).cast<double>();
		  for(auto node_1 = 0; node_1 < n_nodes; node_1++){
		    barycenters_weights(node_1) = barycenters_weights(node_1) + this->phi_function_evaluation_(node_ids(node_idx)) * this->phi_function_evaluation_(node_1);
		  }
		  if(seed_based_r_fit_NA_(k,node_ids(node_idx)) == true){ // the evaluation of the current function is missing on that specific node
		    missing_cell = true;
		  }
		}
		barycenters_depth = barycenters_depth / node_ids.size();
		barycenters_weights = barycenters_weights / node_ids.size();

		// Add the elements to the expectations for each node
		if(!missing_cell){// only if all the nodes in the simplex were present, otherwise skip the cell in this integral
		  for(auto node_1 = 0; node_1 < n_nodes; node_1++){
		      if(this->depth_types_(j)==4 ||( this->depth_types_(j)==5 && seed_patches_[node_1].contains(cell_index))){ // Either we are computing DI-SD and we need to compute all the couples, or PDI-SS, and we restrict to the node ringpatch  or ROI, for each node_1
			expectations_at_nodes(k,node_1) = expectations_at_nodes(k,node_1) + barycenters_depth(node_1)*measure;
			expectations_weight(k,node_1) = expectations_weight(k,node_1) + barycenters_weights(node_1)*measure;
		      }
		  }
		}
	      }
	    } // End FEM-0 case
	  }// End cycle over n_train
	  
	  // finally we perform the integral over the first node
	  // Aux variables
	  DVector<double> barycenters_depth;
	  barycenters_depth.resize(n_train); // Depth
	  DVector<double> barycenters_weights;
	  barycenters_weights.resize(n_train); // Denominator weights
	  DVector<bool> missing_cell;
	  missing_cell.resize(n_train);

	  if(this->int_method_ == -1){ // Voronoi case
	    // Perform Voronoi computation for both the numerator and denominator of the DI depth
	    for(auto i = 0; i < n_nodes; i++){ // For each node of the triangulation (and seed of the dual Voronoi tessellation)
	      // extract the measure of the Voronoi cell 
	      double measure = this->voronoi_.cell(i).measure();
	      if((this->voronoi_.local_dim == 2 && this->voronoi_.embed_dim == 3) || (this->voronoi_.local_dim == 3 && this->voronoi_.embed_dim == 3)){ // In the 2.5D and 3D case we resort t external Voronoi measures
		measure  = this->external_voronoi_measures_[i];
	      }

	      // Add the elements to the overall integral
	      for(auto k=0; k < n_train; k++){
		if(!seed_based_r_fit_NA_(k,i)){ // If the function is missing in the cell k, just skip it (non-exisiting in the integral)
		  weight_den(k) = weight_den(k) + expectations_weight(k,i) * measure;
		  IFD_fit_(k,j) = IFD_fit_(k,j) + expectations_at_nodes(k,i) * measure;
		}
	      }
	    }
	  }else{ // FEM-0 case
	    // Perform FEM computation for both the numerator and denominator of DI depth
	    for(typename D::cell_iterator iter = domain_.cells_begin(); iter != domain_.cells_end(); ++iter){ // For each element (triangle or thetahedron in the Triangulation)
	      // initialize
	      barycenters_depth.setZero();
	      barycenters_weights.setZero();
	      missing_cell.setConstant(false);
	
	      // Get the measure of the simplex
	      double measure = iter->measure();
	  
	      // Extract the nodes indices
	      DVector<int> node_ids = iter->node_ids();
	  
	      // Compute the barycenters
	      for(int node_idx = 0; node_idx < node_ids.size(); node_idx++){
		barycenters_depth = barycenters_depth + expectations_at_nodes.col(node_ids(node_idx));
		barycenters_weights = barycenters_weights + expectations_weight.col(node_ids(node_idx));
		for(auto k=0; k < n_train; k++){
		  if(seed_based_r_fit_NA_(k,node_ids(node_idx)) == true){
		    missing_cell(k) = true;
		  }
		}
	      }
	      barycenters_depth = barycenters_depth / node_ids.size();
	      barycenters_weights = barycenters_weights / node_ids.size();
	  
	      // Add the elements to the overall integral
	      for(auto k=0; k < n_train; k++){
		if(missing_cell(k)==false){ // If the function is missing in the cell k, just skip it (non-exisiting in the integral)
		  weight_den(k) = weight_den(k) + barycenters_weights(k) * measure;
		  IFD_fit_(k,j) = IFD_fit_(k,j) + barycenters_depth(k)*measure;
		}
	      }
	    }
	  } // end of if FEM-0

	  for(auto k=0; k < n_train; k++){ // Normalize w.r.t. the weight denominator
	    if(weight_den(k) > 1e-10) 
	      IFD_fit_(k,j) = IFD_fit_(k,j)/ weight_den(k);
	  }

	}// end depth_types_cycle

	return;
      }
      
      void predict() { 
	
	// compute the seed_based representations for pred functions (Voronoi if int_method_ == -1, FEM otherwise)
	this->compute_seed_based_representation_pred();

	bool single_integral_present = false;
	bool double_integral_present = false;
      
	for(auto j = 0 ; j < this->depth_types_.size(); j++){
	  if(this->pred_depth_types_(j) == 1 || this->pred_depth_types_(j) == 2 || this->pred_depth_types_(j) == 3){
	    single_integral_present = true;
	  }
	  if(this->pred_depth_types_(j) == 4 || this->pred_depth_types_(j) == 5){ // DI-Depth, PDI-Depth
	    double_integral_present = true;
	  }
	}

	int n_pred = this->seed_based_r_pred_.rows();
	int n_nodes = this->domain_.n_nodes();
      
	this->IFD_pred_.resize(n_pred, this->pred_depth_types_.size());
	this->mepi_pred_.resize(n_pred);
	this->mhypo_pred_.resize(n_pred);

	// Call the predictors for the single or double integral cases
	if(single_integral_present){
	  this->predict_single_integral_case();
	}
	if(double_integral_present){
	  this->predict_double_integral_case();
	}

	return;
      }

      void predict_single_integral_case(){
	
        int n_pred = this->seed_based_r_pred_.rows();
	int n_nodes = this->domain_.n_nodes();
      
        Depth_Solver solver(this->seed_based_r_fit_, this->seed_based_r_fit_NA_); // this solver uses the Voronoi representations of the fit functions to estimate the empirical measure.
      
	solver.set_pred_data(this->seed_based_r_pred_);
	solver.set_pred_mask(this->seed_based_r_pred_NA_);
      
	DMatrix<double> point_depth;
	DMatrix<double> point_aux;
	
	point_depth.resize(n_pred, this->pred_depth_types_.size()); // this will contain the point depth, computed for each voronoi element, for each element 
	point_aux.resize(n_pred, 2); // this contains the point-computed auxiliary indices, such as MEPI or MHYPO
	
	// initialization
	for (auto i =0; i < n_pred; i++){
	  mepi_pred_(i) = 0;
	  mhypo_pred_(i) = 0;
	  point_aux(i,0) = 0;
	  point_aux(i,1) = 0;
	  for(auto j =0 ; j<this->pred_depth_types_.size(); j++){
	    IFD_pred_(i,j)=0;
	    point_depth(i,j)=0;
	  }
 	}
	
	// weighting function denominator
	DVector<double> weight_den;
	
	// initialization 
	weight_den.resize(n_pred);
	for(auto i = 0; i< n_pred; i++){
	  weight_den(i)=0;
	}
        
        if(this->int_method_ == -1){ // Voronoi case 
	  // compute the intergals summing up the effect of each cell
	  for (auto i=0; i<n_nodes; i++){
	  
	    double measure =  this->voronoi_.cell(i).measure();
	  
	    if((this->voronoi_.local_dim == 2 && this->voronoi_.embed_dim == 3) || (this->voronoi_.local_dim == 3 && this->voronoi_.embed_dim == 3)){ // In the 2.5D and 3D case we resort t external Voronoi measures
	      measure  = this->external_voronoi_measures_[i];
	    }
	  
	    // compute /int_{O} phi(q(p)) dp
	    for(auto k =0; k<n_pred; k++){
	      if(!seed_based_r_pred_NA_(k,i)){
		weight_den(k) = weight_den(k) + measure * this->phi_function_evaluation_(i);
	      }
	    }
      
	    for (auto j=0; j<this->pred_depth_types_.size(); j++){
      
	      int type = pred_depth_types_(j);
      
	      switch(type) {
	      case 1: // SD: simplicial univariate depth
		{
		  point_depth.col(j) = solver.compute_SD(i) * this->phi_function_evaluation_(i);
		}
		break;
	    
	      case 2: // FMD
		{
		  point_depth.col(j) = solver.compute_FMD(i);
		}
		break;
	    
	      case 3: // MHRD
		{
		  DMatrix<double> MHRD_solution = solver.compute_HD(i) * this->phi_function_evaluation_(i); // Note: this value IS NOT the real point MHRD. MHRD is overwritten afterwards (due to def).
		  point_depth.col(j) = MHRD_solution.col(0); 
		  point_aux = MHRD_solution.rightCols(2);
            
		  mepi_pred_ = mepi_pred_ + point_aux.col(0)*measure;
		  mhypo_pred_ = mhypo_pred_ + point_aux.col(1)*measure;
		}
		break;
	    
	      default:
		{}
		break;
	      }
      
      
	    }
	  
	    // compute /int_{O} D(X(p), pred_functions)*phi(q(p)) dp
	    IFD_pred_ = IFD_pred_ + point_depth*measure;
      
	  }
	}else{ // FEM P0 case
	
	  // Build storage matrices needed for the FEM P0 computation
	  // # nodes x depth_types.size();
	  DMatrix<double> depths_storage;
	  depths_storage.resize(n_pred, n_nodes * pred_depth_types_.size()); // One column for each node and each type of depth required
	  depths_storage.setZero();
	  DMatrix<double> mepi_storage;
	  mepi_storage.resize(n_pred, n_nodes);
	  mepi_storage.setZero();
	  DMatrix<double> mhypo_storage;
	  mhypo_storage.resize(n_pred, n_nodes);
	  mhypo_storage.setZero();
	
	  // Extract the depths from solver before FEM computation.
	  for (auto i=0; i<n_nodes; i++){
	    for (auto j=0; j<this->pred_depth_types_.size(); j++){
	      switch(pred_depth_types_(j)) { // Basing on the type of depth required
	      case 1: //SD: simplicial univariate depth
		{
		  depths_storage.col(i+j*n_nodes) = solver.compute_SD(i) * this->phi_function_evaluation_(i);
	      
		}
		break;
	      
	      case 2: // FMD
		{
		  depths_storage.col(i+j*n_nodes) = solver.compute_FMD(i);
		}  
		break;
	    
	      case 3: // MHRD
		{
		  DMatrix<double> MHRD_solution = solver.compute_HD(i) * this->phi_function_evaluation_(i); 
		  depths_storage.col(i+j*n_nodes) = MHRD_solution.col(0);
		  point_aux = MHRD_solution.rightCols(2); // Epigraph and Hypograph indices
            
		  mepi_storage.col(i) = point_aux.col(0);
		  mhypo_storage.col(i) = point_aux.col(1);
		}  
		break;
	    
	      default:
		{} 
		break;
	     
	      }
      
      
	    }
	  }
	
	  // Aux variables
	  DMatrix<double> barycenters_depth;
	  barycenters_depth.resize(n_pred,this->pred_depth_types_.size() + 2); // Depths + mepi + mhypo
	  double barycenters_weights = 0;
	  DVector<bool> missing_cell;
	  missing_cell.resize(n_pred);
	
	  // Perform FEM computation for both the numerator and denominator of FEMD
	  for(typename D::cell_iterator iter = domain_.cells_begin(); iter != domain_.cells_end(); ++iter){ // For each element (triangle or thetahedron in the Triangulation)
	    // initialize
	    barycenters_depth.setZero();
	    barycenters_weights = 0;
	    missing_cell.setConstant(false);
	
	    // Get the measure of the simplex
	    double measure = iter->measure();
	  
	    // Extract the nodes indices
	    DVector<int> node_ids = iter->node_ids();
	  
	    // Compute the barycenters
	    for(int node_idx = 0; node_idx < node_ids.size(); node_idx++){
	      for(int j = 0; j < this->pred_depth_types_.size(); j++){
		barycenters_depth.col(j) = barycenters_depth.col(j) + depths_storage.col(node_ids(node_idx) +j*n_nodes);
	      }
	      barycenters_depth.col(this->pred_depth_types_.size()) = barycenters_depth.col(this->pred_depth_types_.size()) + mepi_storage.col(node_ids(node_idx));
	      barycenters_depth.col(this->pred_depth_types_.size()+1) = barycenters_depth.col(this->pred_depth_types_.size()+1) + mhypo_storage.col(node_ids(node_idx));
	      barycenters_weights = barycenters_weights + this->phi_function_evaluation_(node_ids(node_idx));
	      for(auto k=0; k < n_pred; k++){
		if(seed_based_r_pred_NA_(k,node_ids(node_idx)) == true){
		  missing_cell(k) = true;
		}
	      }
	    }
	    barycenters_depth = barycenters_depth / node_ids.size();
	    barycenters_weights = barycenters_weights / node_ids.size();
	  
	    // Add the elements to the overall integral
	    for(auto k=0; k < n_pred; k++){
	      if(missing_cell(k)==false){ // If the function is missing in the cell k, just skip it (non-exisiting in the integral)
		weight_den(k) = weight_den(k) + barycenters_weights * measure;
	  
		for(int j =0; j < this->pred_depth_types_.size(); j++){
		  IFD_pred_(k,j) = IFD_pred_(k,j) + barycenters_depth(k,j)*measure;
		}
		mepi_pred_(k) = mepi_pred_(k) + barycenters_depth.col(this->pred_depth_types_.size())(k)*measure;
		mhypo_pred_(k) = mhypo_pred_(k) + barycenters_depth.col(this->pred_depth_types_.size()+1)(k)*measure;
	      }
	    }
	  }
	
	}
	
	for(auto j=0; j < this->pred_depth_types_.size();j++){ 
	  if(pred_depth_types_(j)==3){// 3=="MHRD" The minimum between epigraph and hipograph indices
	    for(auto k =0; k<n_pred; k++){
	      mepi_pred_(k) = mepi_pred_(k) / weight_den(k);
	      mhypo_pred_(k) = mhypo_pred_(k) / weight_den(k);
	      IFD_pred_(k,j) = std::min(mepi_pred_(k), mhypo_pred_(k));
	    }
	  }else{
	    if(pred_depth_types_(j)==1){ // 1==SD: simplicial univariate depth
	      for(auto k=0; k< n_pred; k++){ // for every functional datum
		IFD_pred_(k,j) = IFD_pred_(k,j) / weight_den(k);
	      }
	    }
	  }
	}
     
	return; 
      }

      void predict_double_integral_case(){
	int n_train = this->seed_based_r_fit_.rows();
	int n_pred = this->seed_based_r_pred_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	Bivariate_Depth_Solver solver; // This solver will compute the multivariate depth for each location couple; Will be recycled for each couple, so to avoid multiple dynamic memory allocation
	DMatrix<double> aux_bivariate_data_fit;
	DMatrix<bool> aux_bivariate_data_mask;
	DMatrix<double> aux_couple;
	DMatrix<bool> aux_couple_mask;
	aux_bivariate_data_fit.resize(n_train,2);
	aux_bivariate_data_mask.resize(n_train,2);
	aux_couple.resize(1,2);
	aux_couple_mask.resize(1,2);

	// weighting function denominator
	DVector<double> weight_den; // We will need to fill this while performing the integrals
	
	// Auxiliary variables to store the bivariate depths
	DMatrix<float> depths_storage;
	depths_storage.resize(n_nodes, n_nodes); // One column for each node and each type of depth required // To be initialized in each cycle for functional datum in pred
	
	// Cycle over the required depth types; here the logic differs from the one of the single integral case
	// Compute the depth for each double integral depth type and store it in IFD_pred_ 
	for(auto j = 0; j < this->pred_depth_types_.size(); j++){
	  if(this->pred_depth_types_(j)==1 || this->pred_depth_types_(j)==2|| this->pred_depth_types_(j)==3){ // It is not a double integral, skip it
	    continue; 
	  }
	  
	  // initialization of weight den
	  weight_den.resize(n_pred);
	  for(auto k = 0; k < n_pred; k++){
	    weight_den(k)=0;
	    IFD_pred_(k,j)=0;
	  }
	  
	  // Matrix that stores the result of the second integral over the domain for each node, and for each pred function
	  DMatrix<double> expectations_at_nodes;
	  expectations_at_nodes.resize(n_pred, n_nodes);
	  expectations_at_nodes.setConstant(0);
	  
	  // Matrix that stores the result of the second integral for the weight function over the domain for each node, and for each pred function
	  DMatrix<double> expectations_weight;
	  expectations_weight.resize(n_pred, n_nodes);
	  expectations_weight.setConstant(0);
	  
	  // for each functional datum, store the multivariate depths; then compute the second integral and store the result for each node into Expectation_at_nodes
	  for(auto k = 0; k < n_pred; k++){
	    // initialize depth_storage to -1 (unfeasible value) in order to understand if we have computed a bivariate depth already
	    depths_storage.setConstant(0);

	    // Compute and store the bivariate depths needed
	    for(auto node_1 = 0; node_1 < n_nodes; node_1++){
	      for(auto node_2 = 0; node_2 <= node_1; node_2++){
		if(this->pred_depth_types_(j)==4 ||( this->pred_depth_types_(j)==5 && seed_rings_[node_1].contains(node_2))){ // Either we are computing DI-SD and we need to compute all the couples, or PDI-SS, and we restrict to the node ring or ROI, for each node_1
		  if(!seed_based_r_pred_NA_(k,node_1) && !seed_based_r_pred_NA_(k,node_2)){ // Both the functional evaluations are present, we can compute the multivariate depth with respect to the nonmissing data
		    // set the couple of which we need to compute the multivariate depth
		    aux_couple(0,0) = seed_based_r_pred_(k,node_1);
		    aux_couple(0,1) = seed_based_r_pred_(k,node_2);
		    aux_couple_mask(0,0) = seed_based_r_pred_NA_(k,node_1);
		    aux_couple_mask(0,1) = seed_based_r_pred_NA_(k,node_2);

		    // Set the corresponding bivariate fit data with respect to which we compute the multivariate depth
		    aux_bivariate_data_fit.col(0) = seed_based_r_fit_.col(node_1);
		    aux_bivariate_data_fit.col(1) = seed_based_r_fit_.col(node_2);
		    aux_bivariate_data_mask.col(0) = seed_based_r_fit_NA_.col(node_1);
		    aux_bivariate_data_mask.col(1) = seed_based_r_fit_NA_.col(node_2);
		  
		    // Set the mutlivariate depth solver
		    solver.set_fit_data(aux_bivariate_data_fit, aux_bivariate_data_mask);
		    solver.set_pred_data(aux_couple, aux_couple_mask);

		    // Compute and store the multivariate depth using the bivariate solver
		    depths_storage(node_1, node_2) = solver.compute_SD()(0) * this->phi_function_evaluation_(node_1) * this->phi_function_evaluation_(node_2) ;
		    depths_storage(node_2, node_1) = depths_storage(node_1, node_2); // The multivariate depth is symmetric
		  }// Otherwise just skip the couple, will allso be skipped in the integral
		}
		
	      }
	    }

	    // Second integral computation
	    if(this->int_method_ == -1){ // Voronoi case
	      // Perform Voronoi computation for both the numerator and denominator of the DI depth
	      for(auto node_1 = 0; node_1 < n_nodes; node_1++ ){ // For each node of the triangulation (and seed of the dual Voronoi tessellation)
		for(auto node_2 = 0; node_2 < n_nodes; node_2++ ){ // For each node of the triangulation (and seed of the dual Voronoi tessellation)
		  if(this->pred_depth_types_(j)==4 ||( this->pred_depth_types_(j)==5 && seed_rings_[node_1].contains(node_2))){ // Either we are computing DI-SD and we need to compute all the couples, or PDI-SS, and we restrict to the node ring or ROI, for each node_1
		    // Add the elements to the expectations for each node
		    if(!seed_based_r_pred_NA_(k,node_1) && !seed_based_r_pred_NA_(k,node_2)){// only if all the nodes in the simplex were present, otherwise skip the cell in this integral
		      // extract the measure of the Voronoi cell 
		      double measure = this->voronoi_.cell(node_2).measure();
		      if((this->voronoi_.local_dim == 2 && this->voronoi_.embed_dim == 3) || (this->voronoi_.local_dim == 3 && this->voronoi_.embed_dim == 3)){ // In the 2.5D and 3D case we resort t external Voronoi measures
			measure  = this->external_voronoi_measures_[node_2];
		      }
		      expectations_at_nodes(k,node_1) = expectations_at_nodes(k,node_1) + depths_storage(node_1,node_2)*measure;
		      expectations_weight(k,node_1) = expectations_weight(k,node_1) + this->phi_function_evaluation_(node_1) * this->phi_function_evaluation_(node_2) * measure;
		    }
		  }
		}
	      }
	    }else{ // FEM-0 case
	      // Aux variables that sum up in the expectations
	      DVector<double> barycenters_depth;
	      barycenters_depth.resize(n_nodes); 
	      DVector<double> barycenters_weights;
	      barycenters_weights.resize(n_nodes);
	      bool missing_cell = false;
	      int cell_index = 0;
	
	      // Perform FEM computation for both the numerator and denominator of the DI depth
	      for(typename D::cell_iterator iter = domain_.cells_begin(); iter != domain_.cells_end(); ++iter, ++cell_index){ // For each element (triangle or thetahedron in the Triangulation)
		// initialize
		barycenters_depth.setZero();
		barycenters_weights.setZero();
		missing_cell=false;
	
		// Get the measure of the simplex
		double measure = iter->measure();
	  
		// Extract the nodes indices
		DVector<int> node_ids = iter->node_ids();
	  
		// Compute the barycenters
		for(int node_idx = 0; node_idx < node_ids.size(); node_idx++){
		  barycenters_depth = barycenters_depth + (depths_storage.col(node_ids(node_idx))).cast<double>();
		  for(auto node_1 = 0; node_1 < n_nodes; node_1++){
		    barycenters_weights(node_1) = barycenters_weights(node_1) + this->phi_function_evaluation_(node_ids(node_idx)) * this->phi_function_evaluation_(node_1);
		  }
		  if(seed_based_r_pred_NA_(k,node_ids(node_idx)) == true){ // the evaluation of the current function is missing on that specific node
		    missing_cell = true;
		  }
		}
		barycenters_depth = barycenters_depth / node_ids.size();
		barycenters_weights = barycenters_weights / node_ids.size();

		// Add the elements to the expectations for each node
		if(!missing_cell){// only if all the nodes in the simplex were present, otherwise skip the cell in this integral
		  for(auto node_1 = 0; node_1 < n_nodes; node_1++){
		      if(this->pred_depth_types_(j)==4 ||( this->pred_depth_types_(j)==5 && seed_patches_[node_1].contains(cell_index))){ // Either we are computing DI-SD and we need to compute all the couples, or PDI-SS, and we restrict to the node ringpatch  or ROI, for each node_1
			expectations_at_nodes(k,node_1) = expectations_at_nodes(k,node_1) + barycenters_depth(node_1)*measure;
			expectations_weight(k,node_1) = expectations_weight(k,node_1) + barycenters_weights(node_1)*measure;
		      }
		  }
		}
	      }
	    } // End FEM-0 case
	  }// End cycle over n_train
	  
	  // finally we perform the integral over the first node
	  // Aux variables
	  DVector<double> barycenters_depth;
	  barycenters_depth.resize(n_pred); // Depth
	  DVector<double> barycenters_weights;
	  barycenters_weights.resize(n_pred); // Denominator weights
	  DVector<bool> missing_cell;
	  missing_cell.resize(n_pred);

	  if(this->int_method_ == -1){ // Voronoi case
	    // Perform Voronoi computation for both the numerator and denominator of the DI depth
	    for(auto i = 0; i < n_nodes; i++){ // For each node of the triangulation (and seed of the dual Voronoi tessellation)
	      // extract the measure of the Voronoi cell 
	      double measure = this->voronoi_.cell(i).measure();
	      if((this->voronoi_.local_dim == 2 && this->voronoi_.embed_dim == 3) || (this->voronoi_.local_dim == 3 && this->voronoi_.embed_dim == 3)){ // In the 2.5D and 3D case we resort t external Voronoi measures
		measure  = this->external_voronoi_measures_[i];
	      }

	      // Add the elements to the overall integral
	      for(auto k=0; k < n_pred; k++){
		if(!seed_based_r_pred_NA_(k,i)){ // If the function is missing in the cell k, just skip it (non-exisiting in the integral)
		  weight_den(k) = weight_den(k) + expectations_weight(k,i) * measure;
		  IFD_pred_(k,j) = IFD_pred_(k,j) + expectations_at_nodes(k,i) * measure;
		}
	      }
	    }
	  }else{ // FEM-0 case
	    // Perform FEM computation for both the numerator and denominator of DI depth
	    for(typename D::cell_iterator iter = domain_.cells_begin(); iter != domain_.cells_end(); ++iter){ // For each element (triangle or thetahedron in the Triangulation)
	      // initialize
	      barycenters_depth.setZero();
	      barycenters_weights.setZero();
	      missing_cell.setConstant(false);
	
	      // Get the measure of the simplex
	      double measure = iter->measure();
	  
	      // Extract the nodes indices
	      DVector<int> node_ids = iter->node_ids();
	  
	      // Compute the barycenters
	      for(int node_idx = 0; node_idx < node_ids.size(); node_idx++){
		barycenters_depth = barycenters_depth + expectations_at_nodes.col(node_ids(node_idx));
		barycenters_weights = barycenters_weights + expectations_weight.col(node_ids(node_idx));
		for(auto k=0; k < n_pred; k++){
		  if(seed_based_r_pred_NA_(k,node_ids(node_idx)) == true){
		    missing_cell(k) = true;
		  }
		}
	      }
	      barycenters_depth = barycenters_depth / node_ids.size();
	      barycenters_weights = barycenters_weights / node_ids.size();
	  
	      // Add the elements to the overall integral
	      for(auto k=0; k < n_pred; k++){
		if(missing_cell(k)==false){ // If the function is missing in the cell k, just skip it (non-exisiting in the integral)
		  weight_den(k) = weight_den(k) + barycenters_weights(k) * measure;
		  IFD_pred_(k,j) = IFD_pred_(k,j) + barycenters_depth(k)*measure;
		}
	      }
	    }
	  } // end of if FEM-0

	  for(auto k=0; k < n_pred; k++){ // Normalize w.r.t. the weight denominator
	    if(weight_den(k) > 1e-10) 
	      IFD_pred_(k,j) = IFD_pred_(k,j)/ weight_den(k);
	  }

	}// end depth_types_cycle

	return;
      }
      
    private:
      // Domain handling
      SpaceDomainType domain_;          	        // Triangulated spatial domain
      VoronoiTessellation voronoi_;                     // Voronoi representation of the domain, set in init
      
      // Problem data
      std::vector<DMatrix<double>> locations_; 		// Locations, vector of matrices of locations, dimension may vary. If only one element is present, those are common locations for all the functions; Otherwise we have a locations set for every function. If int_method_ == -1 (FEM) then we have only one matrix that concided with the matrix of nodes 
      std::vector<DMatrix<double>> locations_pred_;    // Locations for predicted functions, vector of matrices of locations, dimension may vary. If only one element is present, those are common locations for all the functions; Otherwise we have a locations set for every function.
      DVector<int> roi_; 	         	        // Vector of integers  indicating the nodes/elements in the Region Of Interest for the Partial Double Integral depth
      std::vector<std::unordered_set<int>> seed_patches_;// A vector of unordered_set, where each element correspond to the  three_node_patch used for the Partial Double Integral
       std::vector<std::unordered_set<int>> seed_rings_;// A vector of unordered_set, where each element correspond to the three_node_ring  used for the Partial Double Integral
      DVector<int> depth_types_; 		        // Vector of strings indicating the types of univariate depths used to compute IFDs required by the user
      DVector<int> pred_depth_types_; 		        // Vector of strings indicating the types of univariate depths used to compute predictive IFDs required by the user
      std::vector<DVector<double>> train_functions_; 	// Functional data used to compute the empirical measures and the associated IFDs, with respect to themeselves. Dimension: n_train, each fucntion may have different size
      std::vector<DVector<bool>> train_matrix_NA_;      // Missing data pattern of the fit functions, used to compute the empirical densisty of the observational process. Dimension n_tain, each mask may have different size
      std::vector<DVector<double>> pred_functions_; 	// Functional data on which will be computed the IFDs with respect to the train functions. Dimension: n_pred, each function may have different size
      std::vector<DVector<bool>> pred_matrix_NA_;       // Missing data pattern of the pred functions, used to compute the empirical densisty of the observational process. Dimension n_pred, each mask may have a different size
      DVector<double> phi_function_evaluation_; 	// Evaluation of the phi function produced in R. Is filled only after the initialization of the model. Size: n_nodes
      DVector<double> external_voronoi_measures_; 	// Measures of the voronoi cells associated to each node
      int int_method_;					// Flag for the type of integration method used. 0 indicates FEM (with integration formula P0 at the momoent)
      
      // Internal data
      DVector<double> observation_density_vector_; 	// Estimated density of the observational process in the Voronoi cells. Is filled after init() has been called. Dimension n_train x n_nodes
      DMatrix<double> seed_based_r_fit_; 	        // Voronoi (int_method_ == -1) / nodes (int_methods == 0)  values for the fit functions. Dimension: n_train x n_nodes
      DMatrix<bool> seed_based_r_fit_NA_;               // Missing data pattern for the Voronoi (int_method_ == -1) / nodes (int_methods == 0) rep. of the train functions. Dimension n_tain x n_nodes
      DMatrix<double> seed_based_r_pred_; 		// Voronoi (int_method_ == -1) / nodes (int_methods == 0) values for the predict functions. Dimension: n_pred x n_nodes
      DMatrix<bool> seed_based_r_pred_NA_;              // Missing data pattern for the Voronoi (int_method_ == -1) / nodes (int_methods == 0)  rep. of the pred functions. Dimension n_pred x n_nodes
      
      // Output
      DMatrix<double> IFD_fit_; 			// Integrated functional depth for fit functions. Dimension: n_train x depth_types.size()
      DMatrix<double> IFD_pred_; 			// Integrated functional depth for predict functions. Dimension: n_pred x pred_depth_types.size()
      DVector<double> mepi_fit_; 			// Modified Epigraph index fit functions. Filled only if MHRD has beed required for fit functions. Size: n_train
      DVector<double> mhypo_fit_; 			// Modified Hypograph index for fit functions. Filled only if MHRD has beed required for fit functions. Size: n_train
      DVector<double> mepi_pred_; 			// Modified Epigraph index pred functions. Filled only if MHRD has beed required for pred functions. Size: n_pred
      DVector<double> mhypo_pred_; 			// Modified Hypograph index for pred functions. Filled only if MHRD has beed required for pred functions. Size: n_pred
      
      // Boxplots components 
      DMatrix<double> medians_; 			// Collection of medians w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<bool> medians_NA_; 			// Collection of medians NA masks w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<double> first_quartile_; 	                // Collection of first_quartile w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<bool> first_quartile_NA_; 		// Collection of first_quartile NA masks w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<double> third_quartile_; 		        // Collection of third_quartile w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<bool> third_quartile_NA_; 		// Collection of third_quartile NA masks w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<double> up_whisker_; 		        // Collection of up_whisker w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<bool> up_whisker_NA_; 			// Collection of up_whisker NA masks w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<double> low_whisker_; 		        // Collection of low_whisker w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<bool> low_whisker_NA_; 			// Collection of low_whisker NA masks w.r.t. the Depths Types requested. Dimension: n_nodes x depth_types.size()
      DMatrix<bool> outliers_;                          // Collection of outliers boolean values (in C++ notation) w.r.t. Depths Types requested. Dimension: n_nodes x depth_types.size()
      
      // initialization methods
      void compute_seed_based_representation_fit(){ 
      
	int n_train = this->train_functions_.size();
	//int n_loc = this->locations_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	if(this->int_method_ == -1){ // Voronoi representation: we need to compute the spatial averages in each voronoi cell.
	
	  // locate the locations (union of the single functions locations) with respect to the voronoi cells
	  std::vector<DVector<int>> locations_in_cells;
	  locations_in_cells.resize(n_train);
	  if(locations_.size() == 1){ // only one locations set: all the functions are referring to the same locations vector (i.e. train_functions is actually a matrix). 
	    DVector<int> locate_out = voronoi_.locate(locations_[0]);
	    for(int i = 0; i < n_train; i++){
	      locations_in_cells[i] = locate_out; // Locate the locations once for all
	    }
	  }else{ // Each function has its own locations set. We are forced to locate every function set
	    for(int i = 0; i < n_train; i++){
	      locations_in_cells[i] = voronoi_.locate(locations_[i]); // Locate the locations for the i-th function
	    }
	  }
	
	
	  // create the matrices that will store the number of locations with non-missimng measure in each location (to be filled in each cycle)
	  DMatrix<int> Count_Train_cells;
	
	  // resize the matrices that will store the voronoi coefficients for train and pred functions
	  seed_based_r_fit_.resize(n_train, n_nodes);
	  seed_based_r_fit_NA_.resize(n_train, n_nodes);
	  Count_Train_cells.resize(n_train, n_nodes);
	
	  // initialization
	  for(auto i = 0; i< n_train; i++){
	    for(auto j =0; j< n_nodes; j++){  
	      seed_based_r_fit_(i,j) = 0;
	      seed_based_r_fit_NA_(i,j) = true;
	      Count_Train_cells(i,j) = 0;
	    }
	  }
	
	  int aux_index;
	
	  // filling the coefficients matrices for fit functions
	  for (auto i =0; i< n_train; i++){
	    int n_loc = locations_in_cells[i].size();
	    for(auto j=0; j< n_loc; j++){
	      if(!train_matrix_NA_[i](j)){
		aux_index = locations_in_cells[i](j);
		Count_Train_cells(i,aux_index)++;
		seed_based_r_fit_NA_(i,aux_index) = false;
		seed_based_r_fit_(i,aux_index) = seed_based_r_fit_(i,aux_index) + train_functions_[i](j);
	      }
	    }
	  }
	
	  // each value is the average of the observed values of the function in the Voronoi cell
	  for(auto i = 0; i< n_train; i++){
	    for(auto j =0; j< n_nodes; j++){  
	      if(Count_Train_cells(i,j)!=0){
		seed_based_r_fit_(i,j) = seed_based_r_fit_(i,j)/Count_Train_cells(i,j);
	      }
	    }
	  }
	
	}else{ // int_method_ = 0: we are using a FEM based representation
	  
	  // In this situation, we have that locations are identical for all the functions, and eventually are the nodes of the Triangulation of the domain.
	  // resize the matrices that will store the voronoi coefficients for train and pred functions
	  seed_based_r_fit_.resize(n_train, n_nodes);
	  seed_based_r_fit_NA_.resize(n_train, n_nodes);
	
	  for(auto i = 0; i< n_train; i++){
	    for(auto j =0; j< n_nodes; j++){
	      // initialization
	      seed_based_r_fit_(i,j) = 0;
	      seed_based_r_fit_NA_(i,j) = true;  
	      if(!train_matrix_NA_[i](j)){
		seed_based_r_fit_(i,j) = train_functions_[i](j);
		seed_based_r_fit_NA_(i,j) = false; 
	      }
	    }
	  }
	}
	
	return;
      }
    
      void compute_seed_based_representation_pred(){
	
	int n_pred = this->pred_functions_.size();
	int n_nodes = this->domain_.n_nodes();
	
	if(this->int_method_ == -1){ // Voronoi based integration: we need to compute the spatial averages
	
	  // locate the locations with respect to the voronoi cells
	  std::vector<DVector<int>> locations_in_cells;
	  locations_in_cells.resize(n_pred);
	  if(locations_pred_.size() == 1){ // only one locations set: all the functions are referring to the same locations vector (i.e. train_functions is actually a matrix). 
	    DVector<int> locate_out = voronoi_.locate(locations_pred_[0]);
	    for(int i = 0; i < n_pred; i++){
	      locations_in_cells[i] = locate_out; // Locate the locations once for all
	    }
	  }else{ // Each function has its own locations set. We are forced to locate every function set
	    for(int i = 0; i < n_pred; i++){
	      locations_in_cells[i] = voronoi_.locate(locations_pred_[i]); // Locate the locations for the i-th function
	    }
	  }
	
	  // create the matrices that will store the number of locations with non-missimng measure in each location (to be filled in each cycle)
	  DMatrix<int> Count_Pred_cells;
	
	  // resize the matrices that will store the voronoi coefficients for train and pred functions
	  seed_based_r_pred_.resize(n_pred, n_nodes);
	  seed_based_r_pred_NA_.resize(n_pred, n_nodes);
	  Count_Pred_cells.resize(n_pred, n_nodes);
	
	  // initialization
	  for(auto i = 0; i< n_pred; i++){
	    for(auto j =0; j< n_nodes; j++){
	      seed_based_r_pred_(i,j) = 0;
	      seed_based_r_pred_NA_(i,j) = true;
	      Count_Pred_cells(i,j) = 0;
	    }
	  }
	
	  int aux_index;
	
	  // Filling the coefficients matrices for pred functions
	  for (auto i =0; i< n_pred; i++){
	    int n_loc = locations_in_cells[i].size();
	    for(auto j=0; j< n_loc; j++){
	      if(!pred_matrix_NA_[i](j)){
		aux_index = locations_in_cells[i](j);
		Count_Pred_cells(i,aux_index)++;
		seed_based_r_pred_NA_(i,aux_index) = false;
		seed_based_r_pred_(i,aux_index) = seed_based_r_pred_(i,aux_index) + pred_functions_[i](j);
	      }
	    }
	  }
	
	  // each value is the average of the observed values of the function in the Voronoi cell
	  for(auto i = 0; i< n_pred; i++){
	    for(auto j =0; j< n_nodes; j++){  
	      if(Count_Pred_cells(i,j)!=0){
		seed_based_r_pred_(i,j) = seed_based_r_pred_(i,j)/Count_Pred_cells(i,j);
	      }
	    }
	  }
	
	}else{ // int_method_==0 --> FEM based representaion, locations is just one single matrix identical with the triangulation nodes
	  // resize the matrices that will store the FEM coefficients for pred functions
	  seed_based_r_pred_.resize(n_pred, n_nodes);
	  seed_based_r_pred_NA_.resize(n_pred, n_nodes);
	
	  // Filling the coefficients matrices for pred functions
	  for (auto i =0; i< n_pred; i++){
	    for(auto j=0; j < n_nodes; j++){
	      if(!pred_matrix_NA_[i](j)){
		seed_based_r_pred_(i,j) = pred_functions_[i](j);
		seed_based_r_pred_NA_(i,j) = false;
	      }else{
	        seed_based_r_pred_(i,j) = 0;
	        seed_based_r_pred_NA_(i,j) = true;
	      }
	    }
	  }
	}
	
	return;
      }

      void compute_seed_patches(){
	
	int n_nodes = seed_based_r_fit_.cols(); // This happens after the seed based representation
	
	this->seed_patches_.resize(seed_based_r_fit_.cols());
	this->seed_rings_.resize(seed_based_r_fit_.cols());

	
	bool partial_double_integral_required=false;
	for(auto d_t : this->depth_types_){
	  if(d_t==5){
	    partial_double_integral_required =true;
	  }
	}
	
	if(!partial_double_integral_required){
	  return; // No partial double integral depth requires, just skip the task
	}

	if(roi_.size()==1){
	  for(auto j = 0; j < n_nodes; ++j){
	    std::vector<int> node_k_ring = domain_.node_k_ring(j,3); // Compute the node_three_ring
	    seed_rings_[j] = std::unordered_set<int>(node_k_ring.begin(),node_k_ring.end());
	    if(this->int_method_!= -1){ // FEM case, we need to compute the node patches
	      std::vector<int> node_k_patch =  domain_.node_k_patch(j,3); // Compute the node_three_patch
	      seed_patches_[j] = std::unordered_set<int>(node_k_patch.begin(),node_k_patch.end()); 
	    }
	  }
	}else{
	  if(this->int_method_ == -1){ // Voronoi case, roi_ contains the nodes of the ROI and we do not need to do anything
	    std::unordered_set<int> roi_nodes(roi_.data(), roi_.data() + roi_.size());
	    for(auto j = 0; j < n_nodes; ++j){
	      seed_rings_[j] = roi_nodes;
	    }
	  }else{ // FEM-0 case, we need to extract the roi_nodes
	    std::unordered_set<int> roi_patch(roi_.data(), roi_.data() + roi_.size());
	    int index = 0;
	    std::unordered_set<int> roi_nodes;
	    for(typename D::cell_iterator iter = domain_.cells_begin(); iter != domain_.cells_end(); ++iter){ // For each element
	      if(roi_patch.count(index)==1){ // This element is in the ROI patch
		// Extract the nodes indices
		DVector<int> node_ids = iter->node_ids();
		for(auto node_idx : node_ids){
		  roi_nodes.insert(node_idx);
		}
	      }
	      index++;
	    }
	    for(auto j = 0; j < n_nodes; ++j){
	      seed_rings_[j] = roi_nodes; // Compute the node_three_ring
	      seed_patches_[j] = roi_patch; // Compute the node_thre_patch
	    }
	  }
	}

	return;
      }
      
      void compute_functional_boxplot(){
      
	int n_train = seed_based_r_fit_.rows();
	int n_nodes = seed_based_r_fit_.cols();
      
	medians_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	medians_NA_.resize(n_nodes,depth_types_.size());
	first_quartile_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	first_quartile_NA_.resize(n_nodes,depth_types_.size());
	third_quartile_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	third_quartile_NA_.resize(n_nodes,depth_types_.size());
	up_whisker_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	up_whisker_NA_.resize(n_nodes,depth_types_.size());
	low_whisker_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	low_whisker_NA_.resize(n_nodes,depth_types_.size());
	
      
	outliers_.resize(n_train, depth_types_.size());
      
	// initialize outliers matricx
	for(auto i = 0; i< n_train; i++){
	  for(auto j=0; j < depth_types_.size(); j++){
	    outliers_(i,j)=false;
	  }
	}
      
	for (auto j=0; j < depth_types_.size(); j++){
      
	  DVector<double> IFD = IFD_fit_.col(j);
	  DVector<double> IFD_sorted = IFD; 
      
	  // Sort the depths
	  std::sort(IFD_sorted.begin(), IFD_sorted.end());
      
	  double max_depth = IFD_sorted(n_train-1); // maximum depth
	  int middle = std::floor(n_train/2); // index that characterizes the minimum depth of the (little more than) 50% of the functions
	  double middle_depth = IFD_sorted(middle); // minimum depth of the (little more than) central 50% of the functions
 	  DVector<int> central_fun_indexes = DVector<int>::Zero(middle); // vector that will store the indices of the central 50% functions
 
	  int count=0;
      
	  // fill median, identify central functions, initialize the quartiles
	  for(auto i = 0; i<n_train && count < middle; i++){
	    if(IFD(i)>=middle_depth){ // If depth is higher than the threshold, add the function to the central 50% ones
	      //central_block.row(count) = seed_based_r_fit_.row(i);
	      central_fun_indexes(count)=i;
	      count++;
	    }
	    if(IFD(i) == max_depth){
	      medians_.col(j) = seed_based_r_fit_.row(i);
	      medians_NA_.col(j) = seed_based_r_fit_NA_.row(i);
	      first_quartile_.col(j) = seed_based_r_fit_.row(i);
	      first_quartile_NA_.col(j) = seed_based_r_fit_NA_.row(i);
	      third_quartile_.col(j) = seed_based_r_fit_.row(i);
	      third_quartile_NA_.col(j) = seed_based_r_fit_NA_.row(i);
	    }
	  }
      
	  // initialization of quartiles: if median is missing we need to select a random value among the central block ones
	  for(auto i = 0; i < n_nodes; i++){
	    if(medians_NA_(i,j)){ // median was missing, bad initialization (value 0 may be out of the range available)
	      bool found=false;
	      int count=0;
	      while(!found && count<middle){
		if(!seed_based_r_fit_NA_(central_fun_indexes(count), i)){ // The function is not missing in node i
		  first_quartile_(i,j) = seed_based_r_fit_(central_fun_indexes(count),i);
		  first_quartile_NA_(i,j) = false;
		  third_quartile_(i,j) = seed_based_r_fit_(central_fun_indexes(count),i);
		  third_quartile_NA_(i,j) = false;
		  found=true;
		}
		count++;
	      }
	    }
	  }
      
	  // now compute the quartiles using the central block: envelope of the central 50% functions
	  for(auto i = 0; i < n_nodes; i++){
	    for(auto k = 0; k < middle; k++){
	      if(!seed_based_r_fit_NA_(central_fun_indexes(k), i)){ // The datum is not missing in the k-th central function
		if(seed_based_r_fit_(central_fun_indexes(k),i) <= first_quartile_(i,j)){
		  first_quartile_(i,j) = seed_based_r_fit_(central_fun_indexes(k),i);
		  first_quartile_NA_(i,j) = seed_based_r_fit_NA_(central_fun_indexes(k),i);
		}
		if(seed_based_r_fit_(central_fun_indexes(k),i) >= third_quartile_(i,j)){
		  third_quartile_(i,j) = seed_based_r_fit_(central_fun_indexes(k),i);
		  third_quartile_NA_(i,j) = seed_based_r_fit_NA_(central_fun_indexes(k),i);
		}
           
	      }
	    }
	  }
      
	  double IQR;
      
	  // Finally fill the whiskers and check outliers
	  for(auto i = 0; i < n_nodes; i++){
     
	    IQR = third_quartile_(i,j) - first_quartile_(i,j);
	    up_whisker_(i,j) = third_quartile_(i,j) + 1.5*IQR;
	    up_whisker_NA_(i,j) = third_quartile_NA_(i,j);
	    low_whisker_(i,j) = first_quartile_(i,j) - 1.5*IQR;
	    low_whisker_NA_(i,j) = first_quartile_NA_(i,j);
      
	    for(auto k=0; k < n_train; k++){
	      if(!seed_based_r_fit_NA_(k, i)){ // The datum is not missing in the function of interest
		if(seed_based_r_fit_(k,i) < low_whisker_(i,j)){
		  outliers_(k,j) = true;
		}
		if(seed_based_r_fit_(k,i) > up_whisker_(i,j)){
		  outliers_(k,j) = true;
		}
           
	      }
	    }
	  }
	}
      
	return;
      
      }
    };
    
    
  }   // namespace models
}   // namespace fdapde





#endif   // __DEPTH_H__
