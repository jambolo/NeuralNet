/** @file *//********************************************************************************************************

                                                       Neuron.h

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/Neuron.h#5 $

	$NoKeywords: $

 ********************************************************************************************************************/

#pragma once

#include <vector>

/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! A neural network neuron
//
//! Source: Russell S. and Norvig P. 1995. <em>Artificial Intelligence: A Modern Approach</em>. Prentice Hall,
//!			Upper Saddle River, N.J. 567-570

class Neuron
{
	friend std::ostream & operator<<( std::ostream & out, Neuron const & n );
	friend std::istream & operator>>( std::istream & in, Neuron & n );

public:

	//! A vector of weights.
	typedef std::vector< float >	WeightVector;

	//! A vector of inputs.
	typedef std::vector< float >	InputVector;

	//! Default constructor
	Neuron();

	//! Constructor
	Neuron( int nInputs );

	//! Constructor
	Neuron( WeightVector const & aWeights );

	// Destructor
	virtual ~Neuron();

	//! Initializes the neuron.
	void Initialize( WeightVector const & aWeights );

	//! Converts inputs to an output.
	float operator()( InputVector const & aInputs ) const;

	//! Converts inputs to an output (supporting back-propagation).
	float operator()( InputVector const & aInputs, float * pd ) const;

	//! Adjusts the weights for each input.
	void AdjustWeights( InputVector const & aInputs, float e, float rate );

	//! Returns the input weights.
	WeightVector const & GetWeights() const				{ return m_aWeights; };

private:

	//! The activation function.
	float Activation( float x ) const;

	//! The activation function (supporting back-propagation).
	float Activation( float x, float * pd ) const;

	//! The input function.
	float Input( InputVector const & aInputs ) const;

	//! The step function for use as an activation function.
	static float Step( float x );

	//! The sign function for use as an activation function.
	static float Sign( float x );

	//! The sigmoid function for use as an activation function.
	static float Sigmoid( float x );

	//! The sigmoid function for use as an activation function (supporting back-propagation).
	static float Sigmoid( float x, float * pd );

	WeightVector	m_aWeights;		//!< Input weights.
};


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! Inserts a Neuron into a stream
std::ostream & operator<<( std::ostream & out, Neuron const & n );

//! Extracts a Neuron from a stream
std::istream & operator>>( std::istream & in, Neuron & n );


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	aInputs		Inputs
//! @return				Output value.

inline float Neuron::operator()( InputVector const & aInputs ) const
{
	return Activation( Input( aInputs ) );
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	aInputs		Inputs
//! @param	pd			A place to store the derivative.
//! @return				Output value.

inline float Neuron::operator()( InputVector const & aInputs, float * pd ) const
{
	return Activation( Input( aInputs ), pd );
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	x	Combined value of the inputs.
//! @return		Result of the activation function.

inline float Neuron::Activation( float x ) const
{
	return Sigmoid( x );
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	x	Combined value of the inputs.
//! @param	pd	A place to store the derivative.
//! @return		Result of the activation function.

inline float Neuron::Activation( float x, float * pd ) const
{
	return Sigmoid( x, pd );
}
