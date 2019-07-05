/** @file *//********************************************************************************************************

                                                      Neuron.cpp

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/Neuron.cpp#5 $

	$NoKeywords: $

 ********************************************************************************************************************/

#include "Neuron.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>

/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//!
//! @warning Use Neuron::Initialize to initialize a Neuron constructed by the default constructor.

Neuron::Neuron()
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//!
//! @param	nInputs		Number of inputs

Neuron::Neuron( int nInputs )
	: m_aWeights( nInputs, 1.f )
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	aWeights		The weights for the inputs.
//!
//! @note	The number of inputs is implied by the size of the weight vector.

Neuron::Neuron( WeightVector const & aWeights )
	: m_aWeights( aWeights )
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

Neuron::~Neuron()
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! The purpose of the function is to initialize a Neuron constructed using the default constructor.
//!
//! @param	aWeights		The weights for the inputs.
//!
//! @note	The number of inputs is changed to the size of the weight vector.

void Neuron::Initialize( WeightVector const & aWeights )
{
	m_aWeights = aWeights;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//!
//! @param	paInputs	The inputs

float Neuron::Input( InputVector const & aInputs ) const
{
	assert( aInputs.size() == m_aWeights.size() );

	int const	nInputs	= (int)aInputs.size();
	float		input	= 0;

	for ( int i = 0; i < nInputs; i++ )
	{
		input += aInputs[i] * m_aWeights[i];
	}

	return input;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! The weights for each input are adjusted using this formula: <tt>W[i] += aInputs[i] * e * rate</tt>.
//!
//! @param	aInputs		Input values used to compute the value of @a e.
//! @param	e			The error term.
//! @param	rate		The learning rate

void Neuron::AdjustWeights( InputVector const & aInputs, float e, float rate )
{
	assert( aInputs.size() == m_aWeights.size() );

	int const	size	= (int)aInputs.size();

	for ( int i = 0; i < size; i++ )
	{
		m_aWeights[i] += aInputs[i] * e * rate;
	}
}

/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! This function is the step function. If @a x >= 0, the output is 1, otherwise the output is 0.
//!
//! @param	x	The input.
//! @return		A result whose value is 0 or 1.

float Neuron::Step( float x )
{
	return float( x >= 0.f );
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! This function is the sign function. If @a x >= 0, the output is 1, otherwise the output is -1.
//!
//! @param	x	The input.
//! @return		A result whose value is -1 or 1.

float Neuron::Sign( float x )
{
	return ( x >= 0.f ) ? 1.f : -1.f;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! This function is the sigmoid function. The function is continous and has these characteristics:
//!		- @a x = 0, output = .5,
//!		- @a x = -infinity, output is 0
//!		- @a x = +infinity, output is 1
//!
//! @param	x	The input.
//! @return		A result whose range is (0,1).

float Neuron::Sigmoid( float x )
{
	return 1.f / ( 1.f + expf( -x ) );
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! This function is the sigmoid function. In addition to computing the sigmoid function, the derivative is also
//! computed. The function is continous and has these characteristics:
//!		- @a x = 0, output = .5,
//!		- @a x = -infinity, output is 0
//!		- @a x = +infinity, output is 1
//!
//! @param	x	The input.
//! @param	pd	A place to store the derivative.
//! @return		A result whose range is (0,1).

float Neuron::Sigmoid( float x, float * pd )
{
	float const	s	= Sigmoid( x );

	// This formula for the derivative has precision problems as x approaches infinity. A better (though more
	// expensive) formula is s / ( 1 + exp(x) ).

	*pd = s * ( 1.f - s );

	return s;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	out		The output stream.
//! @param	n		The Neuron to output.

std::ostream & operator<<( std::ostream & out, Neuron const & n )
{
	int const	size	= (int)n.m_aWeights.size();

	out << size;

	for ( int i = 0; i < size; i++ )
	{
		out << ' ' << n.m_aWeights[i];
	}

	return out;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	in		The input stream.
//! @param	n		The Neuron to input.

std::istream & operator>>( std::istream & in, Neuron & n )
{
	int	size;

	in >> size;

	n.m_aWeights.resize( size );

	for ( int i = 0; i < size; i++ )
	{
		in >> n.m_aWeights[i];
	}

	return in;
}
