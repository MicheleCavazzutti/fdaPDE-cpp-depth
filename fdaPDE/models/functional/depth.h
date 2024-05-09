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
      //DMatrix<double> MHRD_aux;

      bool are_rankings_computed = false; // NBB In prediction put back to false
      //bool is_MHRD_aux_computed = false;

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
      //Depth_Solver()=default;
      Depth_Solver(const DMatrix<double> & fit_data, const DMatrix<bool> & fit_mask): fit_data_(fit_data), fit_mask_(fit_mask){
	n_train = fit_data_.rows();
	n_nodes = fit_data_.cols();	
      }
      
      void set_pred_data(const DMatrix<double> & pred_data){pred_data_ = pred_data;
	n_pred = pred_data.rows();
	are_rankings_computed = false; // NB we need to reset the flag, because we have new preditive data
      }
      void set_pred_mask(const DMatrix<bool> & pred_mask){pred_mask_ = pred_mask;}
      //void reset_rankings_flag(bool flag){this->are_rankings_computed = flag;}

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
	    result(i,1) = (double) (this->n_train - this->NA_number(j) - rankings(i,j) - 1 )/(double) (this->n_train - this->NA_number(j)); // Hepigraph
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
      using Voronoi_tessellation = fdapde::core::Voronoi<SpaceDomainType>;    	// Voronoi tessellation of the spatial domain
      
      static constexpr int M = SpaceDomainType::local_dimension;      		// Not really required
      static constexpr int N = SpaceDomainType::embedding_dimension;

      DEPTH() = default; // Check
      // constructor that takes as imput the mesh and other tools defined in the models
      DEPTH(const D & domain):domain_(domain){};

      // setters
      void set_locations(const DMatrix<double> &  locations){  locations_ =  locations ; }
      void set_train_functions(const DMatrix<double> &  train_functions){  train_functions_ =  train_functions ; }
      void set_train_NA_matrix(const DMatrix<bool> &  NA_matrix){  train_NA_matrix_ =  NA_matrix ; } // Check that a matrix of bools can be copied inside a BinaryMatrix
      void set_pred_functions( const DMatrix<double> & pred_functions) { pred_functions_ = pred_functions ; }
      void set_pred_NA_matrix(const DMatrix<bool> &  NA_matrix){  pred_NA_matrix_ =  NA_matrix ; } // Check that a matrix of bools can be copied inside a BinaryMatrix
      void set_phi_function_evaluation(const DVector<double> & phi_function_evaluation ) { phi_function_evaluation_ = phi_function_evaluation;} // Set phi evaluation from R ( storing phi function)
      void set_depth_types(const DVector<int> & depth_types ) { depth_types_ = depth_types; }
      void set_pred_depth_types(const DVector<int> & depth_types ) { pred_depth_types_ = depth_types; }
      void set_voronoi_r_fit(const DMatrix<double> &  voronoi_r_fit ) { return voronoi_r_fit_ =  voronoi_r_fit ; } // Functions used for fitting and IFDs computation   
      void set_voronoi_r_pred(const DMatrix<double> &  voronoi_r_pred ) { return voronoi_r_pred_ =  voronoi_r_pred; } // Functions used for IFD prediction
      void set_IFD_fit(const DMatrix<double> & IFD_fit ) { IFD_fit_ = IFD_fit; }
      void set_IFD_pred(const DMatrix<double> & IFD_pred ) { IFD_pred_ = IFD_pred; }

      // getters
      const DMatrix<double> & locations() const { return locations_; }
      const DVector<double> & get_density_vector(){return observation_density_vector_; }
      const DMatrix<double> & train_functions() const { return train_functions_; }
      const DMatrix<bool> & train_NA_pattern() const { return train_NA_matrix_; }
      const DMatrix<double> & pred_functions() const { return pred_functions_; }
      const DMatrix<bool> & pred_NA_pattern() const { return pred_NA_matrix_; }
      const SpaceDomainType & domain() const { return domain_; } // Returns the domain used by the method, from which also locations can be accessed
      const Voronoi_tessellation & voronoi() const { return voronoi_; } // Returns the voronoi tessellation used by the method, from which also locations, cells and measures can be accessed
      const DVector<double> & phi_function_evaluation() const { return phi_function_evaluation_; } // Returns phi function used to evaluate the IFD phi in the nodes of the functions
      const DVector<int> & depth_types() const { return depth_types_; }
      const DVector<int> & pred_depth_types() const { return pred_depth_types_; }
      const DMatrix<double> & voronoi_r_fit() const { return voronoi_r_fit_; } // Functions used for fitting and IFDs computation  
      const DMatrix<bool> & Voronoi_fit_NA() const { return Voronoi_NA_fit_; } 
      const DMatrix<double> & voronoi_r_pred() const { return voronoi_r_pred_; } // Functions used for IFD prediction
      const DMatrix<bool> & Vornoy_pred_NA() const { return Voronoi_NA_pred_; } 
      const DMatrix<double> & IFD_fit() const { return IFD_fit_; }
      const DMatrix<double> & IFD_pred() const { return IFD_pred_; }
      
      
      void init() { // Initialization routine, prepares the environment for the solution of the problem.
	// Compute voroni tessellation of the model (for the moment on <2,2>, <1,1> meshes are available)
	Voronoi_tessellation voronoi(domain_); // In future we will need to understand whether to store this object this way or not, maybe in initializztion of the problem
	this->voronoi_ = voronoi; // Save the object in the internal memory for future use
      
	// At first compute the Voronoi representation of data
	this->compute_voronoi_representation_fit();  // This needs to be done after the voronoi has been computed.
	
	// Now we have available in voronoi_r_fit and Voronoi NA fit the computed Voronoi representation of the matrix.
	// We can compute the empirical distribution (Q(p)) in the voronoi nodes, using the NA pattern. We provide equal weight to each element
	// This cycle needs to be modified in light of the Voronoi syntax
	int n_train = train_functions_.rows();
	observation_density_vector_.resize(voronoi_.n_cells());
	for (auto i=0; i<voronoi_.n_cells(); i++){// for each node of the mesh, count how many times a cell has been observed in the Voronoi mask. 
	  auto obs_element = Voronoi_NA_fit_.col(i);
	  observation_density_vector_(i) = n_train - obs_element.count(); // Check wether this is present or not // Be careful of what is bringing nside the Voronoi_NA_fit part: is it counting the NA or the present ones?
	}
	observation_density_vector_ = observation_density_vector_ / n_train; // Here I should count the present ones!!!
	return; 
      } 
      
      void solve() { //  Compute the integrated depths and the outputs that will be returned (save outputs in a df), fill output 
      
	// Get the useful numbers
	int n_train = this->train_functions_.rows();
	int n_nodes = this->voronoi_.n_cells(); 
	
	Depth_Solver solver(this->voronoi_r_fit_, this->Voronoi_NA_fit_); // This solver uses the Voronoi representations of the fit functions to estimate the empirical measure.
      
	solver.set_pred_data(this->voronoi_r_fit_);
	solver.set_pred_mask(this->Voronoi_NA_fit_);
      
	this->IFD_fit_.resize(n_train, this->depth_types_.size());
	this->aux_fit_.resize(n_train,2);
      
	DMatrix<double> point_depth;
	DMatrix<double> point_aux;
	
	point_depth.resize(n_train, this->depth_types_.size()); // this will contain the point depth, computed for each voronoi element, for each element 
	point_aux.resize(n_train, 2); // this contains the computed point auxiliary indices, such as MEPI or MHYPO
	
	//initialization
	for (auto i =0; i < n_train; i++){
	  aux_fit_(i,0) = 0;
	  aux_fit_(i,1) = 0;
	  point_aux(i,0) = 0;
	  point_aux(i,1) = 0;
	  for(auto j =0 ; j<this->depth_types_.size(); j++){
	    IFD_fit_(i,j)=0;
	    point_depth(i,j)=0;
	  }
 	}
	
	// Weighting function denominator
	DVector<double> weight_den;
	
	// Initialization 
	weight_den.resize(n_train);
	for(auto i = 0; i< n_train; i++){
	  weight_den(i)=0;
	}
      
	for (auto i=0; i<n_nodes; i++){
	  // Extract the measure of the Voronoi cell 
	  double measure = this->voronoi_.cell(i).measure();

	  for(auto k =0; k<n_train; k++){
	    if(!Voronoi_NA_fit_(k,i)){
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
            
		aux_fit_ = aux_fit_ + point_aux*measure;
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
	    // Fill the MHRD column with the minimum between between MEPI and MIPO. Note: check wether min is before or after integral!!!
	    for(auto k=0; k< n_train; k++){ // for every functional datum
	      IFD_fit_(k,j) = std::min(aux_fit_(k,0), aux_fit_(k,1)) / weight_den(k);  // Check that the std::min are appropriate in vector!!
	    }
	  }else{
	    if(depth_types_(j)==1){ // 1==MBD
	      for(auto k=0; k< n_train; k++){ // for every functional datum
		IFD_fit_(k,j) = IFD_fit_(k,j) / weight_den(k);
	      }
	    }
	  }
	}
	
	
	// Now prepare output and return
	// To be filled
      
	return; 
      } 
      
      void predict() {
	// Get the useful numbers
	int n_pred = this->pred_functions_.rows();
	int n_nodes = this->domain_.n_nodes(); 
	
	// COmpute the voronoi representations for pred functions
	this->compute_voronoi_representation_pred();
      
        Depth_Solver solver(this->voronoi_r_fit_, this->Voronoi_NA_fit_); // This solver uses the Voronoi representations of the fit functions to estimate the empirical measure.
      
	solver.set_pred_data(this->voronoi_r_pred_);
	solver.set_pred_mask(this->Voronoi_NA_pred_);
	//solver.reset_rankings_flag(false);
      
	this->IFD_pred_.resize(n_pred, this->pred_depth_types_.size());
	this->aux_pred_.resize(n_pred, 2);
      
	DMatrix<double> point_depth;
	DMatrix<double> point_aux;
	
	point_depth.resize(n_pred, this->pred_depth_types_.size()); // this will contain the point depth, computed for each voronoi element, for each element 
	point_aux.resize(n_pred, 2); // this contains the computed point auxiliary indices, such as MEPI or MHYPO
	
	//initialization
	for (auto i =0; i < n_pred; i++){
	  aux_pred_(i,0) = 0;
	  aux_pred_(i,1) = 0;
	  point_aux(i,0) = 0;
	  point_aux(i,1) = 0;
	  for(auto j =0 ; j<this->pred_depth_types_.size(); j++){
	    IFD_pred_(i,j)=0;
	    point_depth(i,j)=0;
	  }
 	}
	
	// Weighting function denominator
	DVector<double> weight_den;
	
	// Initialization 
	weight_den.resize(n_pred);
	for(auto i = 0; i< n_pred; i++){
	  weight_den(i)=0;
	}
      
	for (auto i=0; i<n_nodes; i++){
	  // Check the notion form Alessandro
	  double measure =  this->voronoi_.cell(i).measure();
	  
	  for(auto k =0; k<n_pred; k++){
	    if(!Voronoi_NA_fit_(k,i)){
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
		DMatrix<double> MHRD_solution = solver.compute_MHRD(i) * this->phi_function_evaluation_(i); // Note: this value IS NOT the real point MHRD. MHRD is defined as the global minimum between the MEPI and MHIPO. So it will overwritten afterwards.
		point_depth.col(j) = MHRD_solution.col(0); 
		point_aux = MHRD_solution.rightCols(2);
            
		aux_pred_ = aux_pred_ + point_aux*measure;
	      }
	      break;
	    
	    default:
	      {}
	      break;
	    }
      
      
	  }
	  
	  IFD_pred_ = IFD_pred_ + point_depth*measure;
      
	}
	
	for(auto j=0; j < this->pred_depth_types_.size();j++){ 
	  if(depth_types_(j)==3){// 3=="MHRD" The minimum between epigraph and hipograph indices
	    for(auto k =0; k<n_pred; k++){
	      IFD_pred_(k,j) = std::min(aux_pred_(k,0), aux_pred_(k,1)) / weight_den(k);  // Check that the std::min are appropriate in vector!!
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
      } // Compute the integrated depths for the prediction functions, used only in predict mode
      
    private:
      SpaceDomainType domain_;          	// triangulated spatial domain
      Voronoi_tessellation voronoi_;                // Voronoi representation of the domain, set in init. This component is only local, and needs to be revised when modifications occur in syntax
      DMatrix<double> locations_; 			// Locations, union of the locstions of the fit (and pred) functions. Dimension N x n_loc
      DMatrix<double> train_functions_; 		// Functional data used to compute the empirical measures and the associated IFDs, with respect to themeselves. Dimension: n_train x n_loc
      DMatrix<bool> train_NA_matrix_;   // Binary matrix containing the missing data pattern derived from the train functions, used to compute the empirical densisty of the observational process. Dimension n_tain x n_loc
      DVector<double> observation_density_vector_; 	// This matrix contains the estimated density of the observational process in the Voronoi cells. Is filled after init() has been called. Dimension n_train x n_nodes
      DMatrix<double> pred_functions_; 			// Functional data on which will be computed the IFDs with respect to the train functions. Do not modify the empirical measures. Dimension: n_pred x n_loc
      DMatrix<bool> pred_NA_matrix_;    // Binary matrix containing the missing data pattern derived from the pred functions, used to compute the empirical densisty of the observational process. Dimension n_tain x n_loc
      DVector<double> phi_function_evaluation_; 	// This matrix contains the evaluation of the phi function produced in R. Is filled only after the initialization of the model
      DVector<int> depth_types_; 		// Vector of strings indicating the types of univariate depths used to compute IFDs required by the user
      DVector<int> pred_depth_types_; 		// Vector of strings indicating the types of univariate depths used to compute predictive IFDs required by the user
      DMatrix<double> voronoi_r_fit_; 			// Voronoi values for the fit functions. Dimension: n_train x n_nodes
      DMatrix<bool> Voronoi_NA_fit_;    // Binary matrix containing the missing data pattern for the voronoi representation of the train functions. Dimension n_tain x n_nodes
      DMatrix<double> voronoi_r_pred_; 			// Voronoi values for the predict functions. Dimension: n_predict x n_nodes
      DMatrix<bool> Voronoi_NA_pred_;   // Binary matrix containing the missing data pattern for the voronoi representation of the pred functions. Dimension n_tain x n_nodes
      DMatrix<double> IFD_fit_; 			// Integrated functional depth for fit functions. Dimension: n_train x univariate_depth_types.size()
      DMatrix<double> IFD_pred_; 			// Integrated functional depth for predict functions. Dimension: n_pred x univariate_depth_types.size()
      DMatrix<double> aux_fit_; 			// Auxiliary depth indices for fit functions. Dimension: n_train x univariate_depth_types.size()
      DMatrix<double> aux_pred_; 			// Auxiliary depth indices depth for predict functions. Dimension: n_pred x univariate_depth_types.size()
      
      // initialization methods
      void compute_voronoi_representation_fit(){
	// Implementation up to 30/04/24, to be updated when new syntax is available
	int n_train = this->train_functions_.rows();
	int n_loc = this->locations_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	// Locate the locations (union of the single functions locations) with respect to the voronoi cells. This step is computationally heavy (store it also for predict??? THINK ABOUT THIS)
	DVector<int> locations_in_cells = voronoi_.locate(locations_);
	
	// Create the matrices that will store the number of locations with non-missimng measure in each location (to be filled in each cycle)
	DMatrix<int> Count_Train_cells;
	
	// Resize the matrices that will store the voronoi coefficients for train and pred functions
	voronoi_r_fit_.resize(n_train, n_nodes);
	Voronoi_NA_fit_.resize(n_train, n_nodes);
	Count_Train_cells.resize(n_train, n_nodes);
	
	// initialization
	for(auto i = 0; i< n_train; i++){
	  for(auto j =0; j< n_nodes; j++){  
	    voronoi_r_fit_(i,j) = 0;
	    Voronoi_NA_fit_(i,j) = true;
	    Count_Train_cells(i,j) = 0;
	  }
	}
	
	int aux_index;
	
	// Filling the coefficients matrices for fit functions
	for (auto i =0; i< n_train; i++){
	  for(auto j=0; j< n_loc; j++){
	    if(!train_NA_matrix_(i,j)){
	      aux_index = locations_in_cells(j);
	      Count_Train_cells(i,aux_index)++;
	      Voronoi_NA_fit_(i,aux_index) = false;
	      voronoi_r_fit_(i,aux_index) = voronoi_r_fit_(i,aux_index) + train_functions_(i,j);
	    }
	  }
	}
	
	// compute the averages
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
	// Implementation up to 30/04/24, to be updated when new syntax is available
	int n_pred = this->pred_functions_.rows();
	int n_loc = this->locations_.rows();
	int n_nodes = this->domain_.n_nodes();
	
	// Locate the locations (union of the single functions locations) with respect to the voronoi cells. This step is computationally heavy
	DVector<int> locations_in_cells = voronoi_.locate(locations_);
	
	// Create the matrices that will store the number of locations with non-missimng measure in each location (to be filled in each cycle)
	DMatrix<int> Count_Pred_cells;
	
	// Resize the matrices that will store the voronoi coefficients for train and pred functions
	voronoi_r_pred_.resize(n_pred, n_nodes);
	Voronoi_NA_pred_.resize(n_pred, n_nodes);
	Count_Pred_cells.resize(n_pred, n_nodes);
	
	// initialization
	for(auto i = 0; i< n_pred; i++){
	  for(auto j =0; j< n_nodes; j++){
	    voronoi_r_pred_(i,j) = 0;
	    Voronoi_NA_pred_(i,j) = true;
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
	      Voronoi_NA_pred_(i,aux_index) = false;
	      voronoi_r_pred_(i,aux_index) = voronoi_r_pred_(i,aux_index) + pred_functions_(i,j);
	    }
	  }
	}
	
	// Compute the averages
	for(auto i = 0; i< n_pred; i++){
	  for(auto j =0; j< n_nodes; j++){  
	    if(Count_Pred_cells(i,j)!=0){
	      voronoi_r_pred_(i,j) = voronoi_r_pred_(i,j)/Count_Pred_cells(i,j);
	    }
	  }
	}
	
	return;
      }
      
    };
    
    
  }   // namespace models
}   // namespace fdapde





#endif   // __DEPTH_H__
