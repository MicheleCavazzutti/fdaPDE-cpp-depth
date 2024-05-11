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
//using fdapde::core::BinaryMatrix; // In future we will represent the matrices of bools with this, it is faster.

#include "../model_macros.h"
#include "../model_traits.h"
#include "../sampling_design.h"

namespace fdapde {
  namespace models {
  
    // DEPTH SOLVER CLASS
    class Depth_Solver {
    private:
      const DMatrix<double> & fit_data_;
      const DMatrix<bool> & fit_mask_;
      DMatrix<double> pred_data_;
      DMatrix<bool> pred_mask_;
      int n_train;
      int n_pred;
      int n_nodes;

      DMatrix<int> rankings;
      DVector<int> NA_number;

      bool are_rankings_computed = false; // NBB In prediction put back to false

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
      Depth_Solver(const DMatrix<double> & fit_data, const DMatrix<bool> & fit_mask): fit_data_(fit_data), fit_mask_(fit_mask){
	n_train = fit_data_.rows();
	n_nodes = fit_data_.cols();	
      }
      
      void set_pred_data(const DMatrix<double> & pred_data){pred_data_ = pred_data;
	n_pred = pred_data.rows();
	are_rankings_computed = false; // NB we need to reset the flag, because we have new preditive data
      }
      void set_pred_mask(const DMatrix<bool> & pred_mask){pred_mask_ = pred_mask;}

      DVector<double> compute_MBD(int j){
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
      
      DMatrix<double> compute_MHRD(int j){
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

    // depth model
    template <typename D> 							// Domain type
    class DEPTH { 								// Interface that will be used in R_Depth.cpp
    public:
      using SpaceDomainType = D;          					// triangulated spatial domain
      using VoronoiTessellation = fdapde::core::Voronoi<SpaceDomainType>;    	// Voronoi tessellation of the spatial domain
      
      static constexpr int M = SpaceDomainType::local_dimension;      		// Not really required
      static constexpr int N = SpaceDomainType::embedding_dimension;

      DEPTH() = default; // Check
      // constructor that takes as imput the mesh and other tools defined in the models
      DEPTH(const D & domain):domain_(domain){};

      // setters
      void set_locations(const DMatrix<double> &  locations){  locations_ =  locations ; }
      void set_depth_types(const DVector<int> & depth_types ) { depth_types_ = depth_types; }
      void set_pred_depth_types(const DVector<int> & depth_types ) { pred_depth_types_ = depth_types; }
      void set_train_functions(const DMatrix<double> &  train_functions){  train_functions_ =  train_functions ; }
      void set_train_NA_matrix(const DMatrix<bool> &  NA_matrix){  train_NA_matrix_ =  NA_matrix ; } 
      void set_pred_functions( const DMatrix<double> & pred_functions) { pred_functions_ = pred_functions ; }
      void set_pred_NA_matrix(const DMatrix<bool> &  NA_matrix){  pred_NA_matrix_ =  NA_matrix ; } 
      void set_phi_function_evaluation(const DVector<double> & phi_function_evaluation ) { phi_function_evaluation_ = phi_function_evaluation;} 
      
      void set_voronoi_r_fit(const DMatrix<double> &  voronoi_r_fit ) { return voronoi_r_fit_ =  voronoi_r_fit ; }
      void set_voronoi_r_pred(const DMatrix<double> &  voronoi_r_pred ) { return voronoi_r_pred_ =  voronoi_r_pred; } 
      
      void set_IFD_fit(const DMatrix<double> & IFD_fit ) { IFD_fit_ = IFD_fit; }
      void set_IFD_pred(const DMatrix<double> & IFD_pred ) { IFD_pred_ = IFD_pred; }

      // getters
      const SpaceDomainType & domain() const { return domain_; }                                   // Returns the domain, from which also locations can be accessed
      const VoronoiTessellation & voronoi() const { return voronoi_; }                             // Returns the voronoi tessellation, from which also locations, cells and measures can be accessed
      
      const DVector<int> & depth_types() const { return depth_types_; }
      const DVector<int> & pred_depth_types() const { return pred_depth_types_; }
      const DMatrix<double> & locations() const { return locations_; }
      const DMatrix<double> & train_functions() const { return train_functions_; }
      const DMatrix<bool> & train_NA_pattern() const { return train_NA_matrix_; }
      const DMatrix<double> & pred_functions() const { return pred_functions_; }
      const DMatrix<bool> & pred_NA_pattern() const { return pred_NA_matrix_; }
      const DVector<double> & phi_function_evaluation() const { return phi_function_evaluation_; } // Returns phi function used to evaluate the IFD phi in the nodes of the functions
      
      const DVector<double> & density_vector(){return observation_density_vector_; }
      const DMatrix<double> & voronoi_r_fit() const { return voronoi_r_fit_; }  
      const DMatrix<bool> & voronoi_fit_NA() const { return voronoi_NA_fit_; } 
      const DMatrix<double> & voronoi_r_pred() const { return voronoi_r_pred_; }
      const DMatrix<bool> & vornoy_pred_NA() const { return voronoi_NA_pred_; } 
      
      const DMatrix<double> & IFD_fit() const { return IFD_fit_; }
      const DMatrix<double> & IFD_pred() const { return IFD_pred_; } 
      const DVector<double> & mepi_fit() const { return mepi_fit_; }
      const DVector<double> & mhypo_fit() const { return mhypo_fit_; }
      const DVector<double> & mepi_pred() const { return mepi_pred_; }
      const DVector<double> & mhypo_pred() const { return mhypo_pred_; }
      
      const DMatrix<double> & medians() const { return medians_;} 			
      const DMatrix<bool> & medians_NA() const { return medians_NA_; } 		
      const DMatrix<double> & first_quartile() const { return first_quartile_; }
      const DMatrix<double> & third_quartile() const { return third_quartile_; } 		        
      const DMatrix<double> & up_whisker() const { return up_whisker_; }	        
      const DMatrix<double> & low_whisker() const { return low_whisker_; } 		        
      const DMatrix<bool> & outliers() const { return outliers_; }                       
      
      
      
      void init() { // Initialization routine, prepares the environment for the solution of the problem.
	// Compute voroni tessellation of the model (for the moment only <2,2>, <1,1> meshes are available)
	VoronoiTessellation voronoi(domain_); 
	this->voronoi_ = voronoi;  // Save the object in the internal memory for future use
      
	// At first compute the Voronoi representation of data; this needs to be done after the voronoi has been computed
	this->compute_voronoi_representation_fit(); 
	
	// Now we have available in voronoi_r_fit and Voronoi NA fit the computed Voronoi representation of the matrix.
	// We can compute the empirical distribution (Q(p)) in the voronoi nodes, using the NA pattern. We provide equal weight to each element
	int n_train = train_functions_.rows();
	observation_density_vector_.resize(voronoi_.n_cells());
	for (auto i=0; i<voronoi_.n_cells(); i++){// for each node of the mesh, count how many times a cell has been observed in the Voronoi mask. 
	  auto obs_element = voronoi_NA_fit_.col(i);
	  observation_density_vector_(i) = n_train - obs_element.count();
	}
	observation_density_vector_ = observation_density_vector_ / n_train;
	return; 
      } 
      
      void solve() { //  Compute the integrated depths and the outputs that will be returned (save outputs in a df), fill output 
      
	int n_train = this->train_functions_.rows();
	int n_nodes = this->voronoi_.n_cells(); 
	
	Depth_Solver solver(this->voronoi_r_fit_, this->voronoi_NA_fit_); // This solver uses the Voronoi representations of the fit functions to estimate the empirical measure.
      
	solver.set_pred_data(this->voronoi_r_fit_);
	solver.set_pred_mask(this->voronoi_NA_fit_);
      
	this->IFD_fit_.resize(n_train, this->depth_types_.size());
	this->mepi_fit_.resize(n_train);
	this->mhypo_fit_.resize(n_train);
      
	DMatrix<double> point_depth;
	DMatrix<double> point_aux;
	
	point_depth.resize(n_train, this->depth_types_.size()); // this will contain the point depth, computed for each voronoi element, for each element 
	point_aux.resize(n_train, 2); // this contains the computed point auxiliary indices, such as MEPI or MHYPO
	
	// initialization
	for (auto i =0; i < n_train; i++){
	  mepi_fit_(i) = 0;
	  mhypo_fit_(i) = 0;
	  point_aux(i,0) = 0;
	  point_aux(i,1) = 0;
	  for(auto j =0 ; j<this->depth_types_.size(); j++){
	    IFD_fit_(i,j)=0;
	    point_depth(i,j)=0;
	  }
 	}
	
	// weighting function denominator
	DVector<double> weight_den;
	
	// initialization 
	weight_den.resize(n_train);
	for(auto i = 0; i< n_train; i++){
	  weight_den(i)=0;
	}
      
	for (auto i=0; i<n_nodes; i++){
	  // extract the measure of the Voronoi cell 
	  double measure = this->voronoi_.cell(i).measure();

	  for(auto k =0; k<n_train; k++){
	    if(!voronoi_NA_fit_(k,i)){
	      weight_den(k) = weight_den(k) + measure * this->phi_function_evaluation_(i);
	    }
	  }
      
	  for (auto j=0; j<this->depth_types_.size(); j++){
      
	    int type = depth_types_(j);
      
	    switch(type) {
	    case 1: //MBD
	      {
		point_depth.col(j) = solver.compute_MBD(i) * this->phi_function_evaluation_(i);
	      
	      }
	      break;
	      
	    case 2: // FMD
	      {
		point_depth.col(j) = solver.compute_FMD(i);
	      }  
	      break;
	    
	    case 3: // MHRD
	      {
		DMatrix<double> MHRD_solution = solver.compute_MHRD(i) * this->phi_function_evaluation_(i); // Note: this value IS NOT the real point MHRD. MHRD is defined as the global minimum between the MEPI and MHIPO. So it will be overwritten afterwards.
		point_depth.col(j) = MHRD_solution.col(0);
		point_aux = MHRD_solution.rightCols(2); // Note: I'm not sure that epigraph and ipograph indices should be wieghted for w (phi/int(phi)). In the future we will need to handle this.
            
		mepi_fit_ = mepi_fit_ + point_aux.col(0)*measure;
		mhypo_fit_ = mhypo_fit_ + point_aux.col(1)*measure;
	      }  
	      break;
	    
	    default:
	      {} 
	      break;
	     
	    }
      
      
	  }
	  
	  IFD_fit_ = IFD_fit_ + point_depth*measure;
      
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
	    if(depth_types_(j)==1){ // 1==MBD
	      for(auto k=0; k< n_train; k++){ // for every functional datum
		IFD_fit_(k,j) = IFD_fit_(k,j) / weight_den(k);
	      }
	    }
	  }
	}
	
	// only for fit functions, one can also compute the functional boxplot quantities. In principle, this may also be done in R, but here is faster
	this->compute_functional_boxplot();

	return; 
      } 
      
      void predict() {

	int n_pred = this->pred_functions_.rows();
	int n_nodes = this->domain_.n_nodes(); 
	
	// compute the voronoi representations for pred functions
	this->compute_voronoi_representation_pred();
      
        Depth_Solver solver(this->voronoi_r_fit_, this->voronoi_NA_fit_); // this solver uses the Voronoi representations of the fit functions to estimate the empirical measure.
      
	solver.set_pred_data(this->voronoi_r_pred_);
	solver.set_pred_mask(this->voronoi_NA_pred_);
      
	this->IFD_pred_.resize(n_pred, this->pred_depth_types_.size());
	this->mepi_pred_.resize(n_pred);
	this->mhypo_pred_.resize(n_pred);
      
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
      
        // compute the intergals summing up the effect of each cell
	for (auto i=0; i<n_nodes; i++){
	  
	  double measure =  this->voronoi_.cell(i).measure();
	  // compute /int_{O} phi(q(p)) dp
	  for(auto k =0; k<n_pred; k++){
	    if(!voronoi_NA_fit_(k,i)){
	      weight_den(k) = weight_den(k) + measure * this->phi_function_evaluation_(i);
	    }
	  }
      
	  for (auto j=0; j<this->pred_depth_types_.size(); j++){
      
	    int type = pred_depth_types_(j);
      
	    switch(type) {
	    case 1: // MBD
	      {
		point_depth.col(j) = solver.compute_MBD(i) * this->phi_function_evaluation_(i);
	      }
	      break;
	    
	    case 2: // FMD
	      {
		point_depth.col(j) = solver.compute_FMD(i);
	      }
	      break;
	    
	    case 3: // MHRD
	      {
		DMatrix<double> MHRD_solution = solver.compute_MHRD(i) * this->phi_function_evaluation_(i); // Note: this value IS NOT the real point MHRD. MHRD is overwritten afterwards (due to def).
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
	  
	  // compute /int_{O} D(X(p), fit_functions)*phi(q(p)) dp
	  IFD_pred_ = IFD_pred_ + point_depth*measure;
      
	}
	
	for(auto j=0; j < this->pred_depth_types_.size();j++){ 
	  if(depth_types_(j)==3){// 3=="MHRD" The minimum between epigraph and hipograph indices
	    for(auto k =0; k<n_pred; k++){
	      mepi_pred_(k) = mepi_pred_(k) / weight_den(k);
	      mhypo_pred_(k) = mhypo_pred_(k) / weight_den(k);
	      IFD_pred_(k,j) = std::min(mepi_pred_(k), mhypo_pred_(k)) / weight_den(k);
	    }
	  }else{
	    if(depth_types_(j)==1){ // 1==MBD
	      for(auto k=0; k< n_pred; k++){ // for every functional datum
		IFD_pred_(k,j) = IFD_pred_(k,j) / weight_den(k);
	      }
	    }
	  }
	}
     
	return; 
      }
      
    private:
      // Domain handling
      SpaceDomainType domain_;          	        // Triangulated spatial domain
      VoronoiTessellation voronoi_;                     // Voronoi representation of the domain, set in init
      
      // Problem data
      DMatrix<double> locations_; 			// Locations, union of the locstions of the fit (and pred) functions. Dimension N x n_loc
      DVector<int> depth_types_; 		        // Vector of strings indicating the types of univariate depths used to compute IFDs required by the user
      DVector<int> pred_depth_types_; 		        // Vector of strings indicating the types of univariate depths used to compute predictive IFDs required by the user
      DMatrix<double> train_functions_; 		// Functional data used to compute the empirical measures and the associated IFDs, with respect to themeselves. Dimension: n_train x n_loc
      DMatrix<bool> train_NA_matrix_;                   // Missing data pattern of the fit functions, used to compute the empirical densisty of the observational process. Dimension n_tain x n_loc
      DMatrix<double> pred_functions_; 			// Functional data on which will be computed the IFDs with respect to the train functions. Dimension: n_pred x n_loc
      DMatrix<bool> pred_NA_matrix_;                    // Missing data pattern of the pred functions, used to compute the empirical densisty of the observational process. Dimension n_pred x n_loc
      DVector<double> phi_function_evaluation_; 	// Evaluation of the phi function produced in R. Is filled only after the initialization of the model. Size: n_nodes
      
      // Internal data
      DVector<double> observation_density_vector_; 	// Estimated density of the observational process in the Voronoi cells. Is filled after init() has been called. Dimension n_train x n_nodes
      DMatrix<double> voronoi_r_fit_; 			// Voronoi values for the fit functions. Dimension: n_train x n_nodes
      DMatrix<bool> voronoi_NA_fit_;                    // Missing data pattern for the voronoi representation of the train functions. Dimension n_tain x n_nodes
      DMatrix<double> voronoi_r_pred_; 			// Voronoi values for the predict functions. Dimension: n_pred x n_nodes
      DMatrix<bool> voronoi_NA_pred_;                   // Missing data pattern for the voronoi representation of the pred functions. Dimension n_pred x n_nodes
      
      // Output
      DMatrix<double> IFD_fit_; 			// Integrated functional depth for fit functions. Dimension: n_train x depth_types.size()
      DMatrix<double> IFD_pred_; 			// Integrated functional depth for predict functions. Dimension: n_pred x pred_depth_types.size()
      DVector<double> mepi_fit_; 			// Modified Epigraph index fit functions. Filled only if MHRD has beed required for fit functions. Size: n_train
      DVector<double> mhypo_fit_; 			// Modified Hypograph index for fit functions. Filled only if MHRD has beed required for fit functions. Size: n_train
      DVector<double> mepi_pred_; 			// Modified Epigraph index pred functions. Filled only if MHRD has beed required for pred functions. Size: n_pred
      DVector<double> mhypo_pred_; 			// Modified Hypograph index for pred functions. Filled only if MHRD has beed required for pred functions. Size: n_pred
      
      // Boxplots components 
      DMatrix<double> medians_; 			// Collection of medians w.r.t. the Depths Types requested. Dimension: n_train x depth_types.size()
      DMatrix<bool> medians_NA_; 			// Collection of medians NA masks w.r.t. the Depths Types requested. Dimension: n_train x depth_types.size()
      DMatrix<double> first_quartile_; 	                // Collection of first_quartile w.r.t. the Depths Types requested. Dimension: n_train x depth_types.size()
      DMatrix<double> third_quartile_; 		        // Collection of third_quartile w.r.t. the Depths Types requested. Dimension: n_train x depth_types.size()
      DMatrix<double> up_whisker_; 		        // Collection of up_whisker w.r.t. the Depths Types requested. Dimension: n_train x depth_types.size()
      DMatrix<double> low_whisker_; 		        // Collection of low_whisker w.r.t. the Depths Types requested. Dimension: n_train x depth_types.size()
      DMatrix<bool> outliers_;                           // Collection of outliers boolean values (in C++ notation) w.r.t. Depths Types requested. Dimension: n_train x depth_types.size()
      
      // initialization methods
      void compute_voronoi_representation_fit(){
      
	int n_train = this->train_functions_.rows();
	int n_loc = this->locations_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	// locate the locations (union of the single functions locations) with respect to the voronoi cells
	DVector<int> locations_in_cells = voronoi_.locate(locations_);
	
	// create the matrices that will store the number of locations with non-missimng measure in each location (to be filled in each cycle)
	DMatrix<int> Count_Train_cells;
	
	// resize the matrices that will store the voronoi coefficients for train and pred functions
	voronoi_r_fit_.resize(n_train, n_nodes);
	voronoi_NA_fit_.resize(n_train, n_nodes);
	Count_Train_cells.resize(n_train, n_nodes);
	
	// initialization
	for(auto i = 0; i< n_train; i++){
	  for(auto j =0; j< n_nodes; j++){  
	    voronoi_r_fit_(i,j) = 0;
	    voronoi_NA_fit_(i,j) = true;
	    Count_Train_cells(i,j) = 0;
	  }
	}
	
	int aux_index;
	
	// filling the coefficients matrices for fit functions
	for (auto i =0; i< n_train; i++){
	  for(auto j=0; j< n_loc; j++){
	    if(!train_NA_matrix_(i,j)){
	      aux_index = locations_in_cells(j);
	      Count_Train_cells(i,aux_index)++;
	      voronoi_NA_fit_(i,aux_index) = false;
	      voronoi_r_fit_(i,aux_index) = voronoi_r_fit_(i,aux_index) + train_functions_(i,j);
	    }
	  }
	}
	
	// each value is the average of the observed values of the function in the Voronoi cell
	for(auto i = 0; i< n_train; i++){
	  for(auto j =0; j< n_nodes; j++){  
	    if(Count_Train_cells(i,j)!=0){
	      voronoi_r_fit_(i,j) = voronoi_r_fit_(i,j)/Count_Train_cells(i,j);
	    }
	  }
	}
	
	return;
      }
    
      void compute_voronoi_representation_pred(){
	
	int n_pred = this->pred_functions_.rows();
	int n_loc = this->locations_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	// locate the locations (union of the single functions locations) with respect to the voronoi cells
	DVector<int> locations_in_cells = voronoi_.locate(locations_);
	
	// create the matrices that will store the number of locations with non-missimng measure in each location (to be filled in each cycle)
	DMatrix<int> Count_Pred_cells;
	
	// resize the matrices that will store the voronoi coefficients for train and pred functions
	voronoi_r_pred_.resize(n_pred, n_nodes);
	voronoi_NA_pred_.resize(n_pred, n_nodes);
	Count_Pred_cells.resize(n_pred, n_nodes);
	
	// initialization
	for(auto i = 0; i< n_pred; i++){
	  for(auto j =0; j< n_nodes; j++){
	    voronoi_r_pred_(i,j) = 0;
	    voronoi_NA_pred_(i,j) = true;
	    Count_Pred_cells(i,j) = 0;
	  }
	}
	
	int aux_index;
	
	// Filling the coefficients matrices for pred functions
	for (auto i =0; i< n_pred; i++){
	  for(auto j=0; j< n_loc; j++){
	    if(!pred_NA_matrix_(i,j)){
	      aux_index = locations_in_cells(j);
	      Count_Pred_cells(i,aux_index)++;
	      voronoi_NA_pred_(i,aux_index) = false;
	      voronoi_r_pred_(i,aux_index) = voronoi_r_pred_(i,aux_index) + pred_functions_(i,j);
	    }
	  }
	}
	
	// each value is the average of the observed values of the function in the Voronoi cell
	for(auto i = 0; i< n_pred; i++){
	  for(auto j =0; j< n_nodes; j++){  
	    if(Count_Pred_cells(i,j)!=0){
	      voronoi_r_pred_(i,j) = voronoi_r_pred_(i,j)/Count_Pred_cells(i,j);
	    }
	  }
	}
	
	return;
      }
      
      void compute_functional_boxplot(){
      
	int n_train = voronoi_r_fit_.rows();
	int n_nodes = voronoi_r_fit_.cols();
      
	medians_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	medians_NA_.resize(n_nodes,depth_types_.size());
	first_quartile_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	third_quartile_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	up_whisker_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	low_whisker_ = DMatrix<double>::Zero(n_nodes, depth_types_.size());
	
      
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
	      //central_block.row(count) = voronoi_r_fit_.row(i);
	      central_fun_indexes(count)=i;
	      count++;
	    }
	    if(IFD(i) == max_depth){
	      medians_.col(j) = voronoi_r_fit_.row(i);
	      medians_NA_.col(j) = voronoi_NA_fit_.row(i);
	      first_quartile_.col(j) = voronoi_r_fit_.row(i); //NO this way we are not handling well the possibly missing 
	      third_quartile_.col(j) = voronoi_r_fit_.row(i);
	    }
	  }
      
	  // initialization of quartiles: if median is missing we need to select a random value among the central block ones
	  for(auto i = 0; i < n_nodes; i++){
	    if(medians_NA_(i,j)){ // median was missing, bad initialization (value 0 may be out of the range available)
	      bool found=false;
	      int count=0;
	      while(!found && count<middle){
		if(!voronoi_NA_fit_(central_fun_indexes(count), i)){ // The function is not missing in node i
		  first_quartile_(i,j) = voronoi_r_fit_(central_fun_indexes(count),i);
		  third_quartile_(i,j) = voronoi_r_fit_(central_fun_indexes(count),i);
		  found=true;
		}
		count++;
	      }
	    }
	  }
      
	  // now compute the quartiles using the central block: envelope of the central 50% functions
	  for(auto i = 0; i < n_nodes; i++){
	    for(auto k = 0; k < middle; k++){
	      if(!voronoi_NA_fit_(central_fun_indexes(k), i)){ // The datum is not missing in the k-th central function
		if(voronoi_r_fit_(central_fun_indexes(k),i) <= first_quartile_(i,j)){
		  first_quartile_(i,j) = voronoi_r_fit_(central_fun_indexes(k),i);
		}
		if(voronoi_r_fit_(central_fun_indexes(k),i) >= third_quartile_(i,j)){
		  third_quartile_(i,j) = voronoi_r_fit_(central_fun_indexes(k),i);
		}
           
	      }
	    }
	  }
      
	  double IQR;
      
	  // Finally fill the whiskers and check outliers
	  for(auto i = 0; i < n_nodes; i++){
     
	    IQR = third_quartile_(i,j) - first_quartile_(i,j);
	    up_whisker_(i,j) = third_quartile_(i,j) + 1.5*IQR;
	    low_whisker_(i,j) = first_quartile_(i,j) - 1.5*IQR;
      
	    for(auto k=0; k < n_train; k++){
	      if(!voronoi_NA_fit_(k, i)){ // The datum is not missing in the function of interest
		if(voronoi_r_fit_(k,i) < low_whisker_(i,j)){
		  outliers_(k,j) = true;
		}
		if(voronoi_r_fit_(k,i) > up_whisker_(i,j)){
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
