/** @file *//********************************************************************************************************

                                                     NeuralNet.h

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/NeuralNet.h#5 $

	$NoKeywords: $

 ********************************************************************************************************************/

#pragma once

#include "Neuron.h"

#include <vector>


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! A neural network interface
//
//! This class is a generic interface for neural net classes.
//!
//! Source: Russell S. and Norvig P. 1995. "Neural Networks" <em>Artificial Intelligence: A Modern Approach</em>.
//!			Prentice Hall, Upper Saddle River, N.J.
//!
//! @note	This is an abstract base class.

class NeuralNet
{
	friend std::ostream & operator<<( std::ostream & out, NeuralNet const & nn );
	friend std::istream & operator>>( std::istream & in, NeuralNet & nn );

public:

	//! A vector of output values.
	typedef std::vector< float >	OutputVector;

	//! A vector of error values.
	typedef std::vector< float >	ErrorVector;

	//! Constructor
	NeuralNet();

	//! Constructor
	//
	//! @param	nInputs		Number of inputs.
	//! @param	nOutputs	Number of outputs
	NeuralNet( int nInputs, int nOutputs );

	//! Destructor
	virtual ~NeuralNet();

	//! Computes an output for the given input.
	//
	//! @param	aInputs		The input values.
	//! @return				A vector of output values. The size of the vector is the number of outputs.

	virtual OutputVector const & operator()( Neuron::InputVector const & aInputs ) = 0;

	//! Trains the system by applying error values.
	//
	//!
	//! @param	aInputs		The input values
	//! @param	aErrors		The error values for each output. The number of error values must be equal to the
	//!						number of output values.
	//! @param	rate		The learning rate.

	virtual void Train( Neuron::InputVector const & aInputs, ErrorVector const & aErrors, float rate ) = 0;

protected:

	int				m_nInputs;				//!< The number of inputs to the net.
	OutputVector	m_aOutputs;				//!< The outputs from most recent set of inputs.
											//!< @note The size of the vector is the number of outputs from the
											//!< net.
};


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! Inserts a NeuralNet into a stream.
std::ostream & operator<<( std::ostream & out, NeuralNet const & nn );

//! Extracts a NeuralNet from a stream.
std::istream & operator>>( std::istream & in, NeuralNet & nn );
