/** @file *//********************************************************************************************************

                                               MultilayerFeedForward.h

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/MultilayerFeedForward.h#3 $

	$NoKeywords: $

 ********************************************************************************************************************/

#pragma once

#include "NeuralNet.h"


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! A multilayer feed-forward neural net with back-propagation
//
//! Source: Russell S. and Norvig P. 1995. "Multilayer Feed-Forward Networks" <em>Artificial Intelligence: A
//!			Modern Approach</em>. Prentice Hall, Upper Saddle River, N.J.

class MultilayerFeedForward : public NeuralNet
{
	friend std::ostream & operator<<( std::ostream & out, MultilayerFeedForward const & mff );
	friend std::istream & operator>>( std::istream & in, MultilayerFeedForward & mff );

public:

	//! Constructor
	MultilayerFeedForward();

	//! Constructor
	MultilayerFeedForward( int nInputs, int nHidden, int nOutputs );

	//! Constructor
	MultilayerFeedForward( int nInputs, int nHidden, int nOutputs, Neuron::WeightVector const & aWeights );

	//! Destructor
	~MultilayerFeedForward();

	//! @name Overrides NeuralNet
	//@{
	virtual OutputVector const & operator()( Neuron::InputVector const & aInputs );
	virtual void Train( Neuron::InputVector const & aInputs, ErrorVector const & aErrors, float rate );
	//@}

private:

	//! A vector of units.
	typedef std::vector< Neuron >	UnitVector;

	//! A vector of gradient values.
	typedef std::vector< float >	GradientVector;

	UnitVector		m_aHiddenUnits;			//!< The array of hidden units.
	OutputVector	m_aHiddenOutputs;		//!< The outputs from the hidden units (inputs to the output units).
	GradientVector	m_aHiddenGradients;		//!< The gradients of the outputs from the hidden units.
	UnitVector		m_aOutputUnits;			//!< The array of output units.
	GradientVector	m_aOutputGradients;		//!< The gradients of the outputs from the output units.
};


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! Inserts a MultilayerFeedForward into a stream.
std::ostream & operator<<( std::ostream & out, MultilayerFeedForward const & mff );

//! Extracts a MultilayerFeedForward from a stream.
std::istream & operator>>( std::istream & in, MultilayerFeedForward & mff );
