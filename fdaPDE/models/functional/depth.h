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
using fdapde::core::Mesh;

#include "../model_macros.h"
#include "../model_traits.h"
#include "../sampling_design.h"

namespace fdapde {
  namespace models {

    // depth model
    template <typename D> 				// Domain type
    class DEPTH { 					// Interface that will be used in R_Depth.cpp
    public:
      using SpaceDomainType = D;          		// triangulated spatial domain
      using Base::df_;                    		// BlockFrame for problem's data storage
      
      static constexpr int M = SpaceDomainType::local_dimension;      	// Not really required
      static constexpr int N = SpaceDomainType::embedding_dimension;

      DEPTH() = default; // Check
      // constructor that takes as imput the mesh and other tools defined in the models
      DEPTH(const D & domain):domain_(domain);

      // setters
      void set_locations(const DMatrix<double> &  locations){  locations_ =  locations ; }
      void set_train_functions(const DMatrix<double> &  train_functions){  train_functions_ =  train_functions ; }
      void set_pred_functions( const DMatrix<double> & pred_functions) { pred_functions_ = pred_functions ; }
      void set_phi_function(std::function phi_function ) { phi_function_ = phi_function; } // Returns phi function used to evaluate the IFD phi in the nodes of the functions
      void set_depth_types(const Dvector<std::string> & depth_types ) { depth_types_ = depth_types; }
      void set_voronoy_r_fit(const DMatrix<double> &  voronoy_r_fit ) { return voronoy_r_fit_ =  voronoy_r_fit ; } // Functions used for fitting and IFDs computation   
      void set_voronoy_r_pred(const DMatrix<double> &  voronoy_r_pred ) { return voronoy_r_pred_ =  voronoy_r_pred; } // Functions used for IFD prediction
      void set_df_fit( const BlockFrame<double> & df_fit ) { df_fit_ = df_fit ; } // We will set something different from R_DEPTH
      void set_df_pred( const BlockFrame<double> & df_pred ) { df_pred_ = df_pred ; } // Contain various auxiliary outputs
      void set_IFD_fit(const DMatrix<double> & IFD_fit ) { IFD_fit_ = IFD_fit; }
      void set_IFD_pred(const DMatrix<double> & IFD_pred ) { IFD_pred_ = IFD_pred; }

      // getters
      const DMatrix<double> & locations() const { return locations_; }
      const DMatrix<double> & get_density_matrix(){return Observation_density_matrix_; }
      const DMatrix<double> & train_functions() const { return train_functions_; }
      const DMatrix<double> & pred_functions() const { return pred_functions_; }
      const SpaceDomainType & domain() const { return domain_; } // Returns the domain used by the method, from which also locations can be accessed
      const DMatrix<double> & phi_function_evaluation() const { return phi_function_evaluation_; } // Returns phi function used to evaluate the IFD phi in the nodes of the functions
      const Dvector<std::string> & depth_types() const { return depth_types_; }
      const DMatrix<double> & voronoy_r_fit() const { return voronoy_r_fit_; } // Functions used for fitting and IFDs computation   
      const DMatrix<double> & voronoy_r_pred() const { return voronoy_r_pred_; } // Functions used for IFD prediction
      const BlockFrame<double> & df_fit() const { return df_fit_; } // Contain various auxiliary outputs
      const BlockFrame<double> & df_pred() const { return df_pred_; } // Contain various auxiliary outputs
      const DMatrix<double> & IFD_fit() const { return IFD_fit_; }
      const DMatrix<double> & IFD_pred() const { return IFD_pred_; }
      
      // initialization methods
      void compute_voronoy_representation_fit(){
      
      
      return;
      }
      void init() { return; } // Initialization routine, prepares the environment for the solution of the problem.
      void solve() { return; } //  Compute the integrated depths and the outputs that will be returned (save outputs in a df), fill output
      void predict() {return; } // Compute the integrated depths for the prediction functions, used only in predict mode
      
    private:
      const SpaceDomainType & domain_;          	// triangulated spatial domain
      DMatrix<double> locations_; 			// Locations, union of the locstions of the fit (and pred) functions. Dimension N x n_loc
      DMatrix<double> Observation_density_matrix_; 	// This matrix contains the estimated density of the observational process in the Voronoy cells. Is filled after init() has been called. Dimension n_train x n_nodes
      DMatrix<double> train_functions_; 		// Functional data used to compute the empirical measures and the associated IFDs, with respect to themeselves. Dimension: n_train x n_loc
      DMatrix<double> pred_functions_; 			// Functional data on which will be computed the IFDs with respect to the train functions. Do not modify the empirical measures. Dimension: n_pred x n_loc
      DMatrix<double> phi_function_evaluation_: 	// This matrix contains the evaluation of the phi function produced in R. Is filled only after the initialization of the model
      DVector<std::string> depth_types_; 		// Vector of strings indicating the types of univariate depths used to compute IFDs required by the user
      DMatrix<double> voronoy_r_fit_; 			// Voronoy values for the fit functions. Dimension: n_train x n_nodes
      DMatrix<double> voronoy_r_pred_; 			// Voronoy values for the predict functions. Dimension: n_predict x n_nodes
      BlockFrame<double> df_fit_;   	   		// blockframe that contains the output for the fit functions. In order: Median, Q1, Q3, UpperFence, LowerFence, MEI, MHO
      BlockFrame<double> df_pred_;      		// blockframe that contains the output for the predict functions. In order: Median, Q1, Q3, UpperFence, LowerFence, MEI, MHO
      DMatrix<double> IFD_fit_; 			// Integrated functional depth for fit functions. Dimension: n_train x univariate_depth_types.size()
      DMatrix<double> IFD_pred_; 			// Integrated functional depth for predict functions. Dimension: n_pred x univariate_depth_types.size()
    };

  }   // namespace models
}   // namespace fdapde

#endif   // __DEPTH_H__
