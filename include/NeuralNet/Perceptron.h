/** @file *//********************************************************************************************************

                                                     Perceptron.h

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/Perceptron.h#5 $

	$NoKeywords: $

 ********************************************************************************************************************/

#pragma once

#include "NeuralNet.h"
#include "Neuron.h"

#include <iosfwd>
#include <vector>

/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! The classic Perceptron neural net.
//
//! The classic Perceptron neural net is a single-layer feed-forward network.
//!
//! Source: Russell S. and Norvig P. 1995. "Perceptrons" <em>Artificial Intelligence: A Modern Approach</em>.
//!			Prentice Hall, Upper Saddle River, N.J.

class Perceptron : public NeuralNet
{
	friend std::ostream & operator<<( std::ostream & out, Perceptron const & p );
	friend std::istream & operator>>( std::istream & in, Perceptron & p );

public:

	//! Constructor
	Perceptron();

	//! Constructor
	Perceptron( int nInputs, int nOutputs );

	//! Constructor
	Perceptron( int nInputs, int nOutputs, Neuron::WeightVector const & aWeights );

	//! Destructor
	~Perceptron();

	//! @name Overrides NeuralNet
	//@{
	virtual OutputVector const & operator()( Neuron::InputVector const & aInputs );
	virtual void Train( Neuron::InputVector const & aInputs, ErrorVector const & aErrors, float rate );
	//@}

private:

	//! A vector of units.
	typedef std::vector< Neuron >	UnitVector;

	UnitVector	m_aOutputUnits;			//!< The array of neurons.
};


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! Inserts a Perceptron into a stream.
std::ostream & operator<<( std::ostream & out, Perceptron const & p );

//! Extracts a Perceptron from a stream.
std::istream & operator>>( std::istream & in, Perceptron & p );
