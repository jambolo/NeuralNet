/** @file *//********************************************************************************************************

                                                    Perceptron.cpp

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/Perceptron.cpp#5 $

	$NoKeywords: $

 ********************************************************************************************************************/

#include "Perceptron.h"

#include <vector>
#include <iostream>
#include <cassert>


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

Perceptron::Perceptron()
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	nInputs		The number of inputs.
//! @param	nOutputs	The number of outputs.

Perceptron::Perceptron( int nInputs, int nOutputs )
	: NeuralNet( nInputs, nOutputs ),
	m_aOutputUnits( nOutputs, Neuron( nInputs ) )
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	nInputs		The number of inputs.
//! @param	nOutputs	The number of outputs.
//! @param	aWeights	The weights for each input of each neuron. There must be ( @a nInputs + 1 ) * @a nOutputs
//!						weights. The additional weight for each output is the threshold value.

Perceptron::Perceptron( int nInputs, int nOutputs, Neuron::WeightVector const & aWeights )
	: NeuralNet( nInputs, nOutputs ),
	m_aOutputUnits( nOutputs )
{
	Neuron::WeightVector::const_iterator	pFirst	= aWeights.begin();

	for ( int i = 0; i < nOutputs; i++ )
	{
		m_aOutputUnits[i].Initialize( Neuron::WeightVector( pFirst, pFirst + nInputs ) );
		pFirst += nInputs;
	}
}

/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

Perceptron::~Perceptron()
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

Perceptron::OutputVector const & Perceptron::operator()( Neuron::InputVector const & aInputs )
{
	assert( m_aOutputs.size() == m_aOutputUnits.size() );

	int const	nOutputs	= (int)m_aOutputs.size();

	for ( int i = 0; i < nOutputs; i++ )
	{
		m_aOutputs[i] = m_aOutputUnits[i]( aInputs );
	}

	return m_aOutputs;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

void Perceptron::Train( Neuron::InputVector const & aInputs, ErrorVector const & aErrors, float rate )
{
	assert( aInputs.size() == m_nInputs );
	assert( aErrors.size() == m_aOutputUnits.size() );

	int const	nOutputs	= (int)m_aOutputUnits.size();

	// Train each neuron.

	for ( int i = 0; i < nOutputs; i++ )
	{
		m_aOutputUnits[i].AdjustWeights( aInputs, aErrors[i], rate );
	}

}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	out		The output stream.
//! @param	p		The Perceptron to output.

std::ostream & operator<<( std::ostream & out, Perceptron const & p )
{
	out << static_cast< NeuralNet const & >( p );

	int const	nOutputs	= (int)p.m_aOutputUnits.size();

	out << ' ' << nOutputs << std::endl;

	for ( int i = 0; i < nOutputs; i++ )
	{
		out << p.m_aOutputUnits[i] << std::endl;
	}

	return out;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	in		The input stream.
//! @param	p		The Perceptron to input.

std::istream & operator>>( std::istream & in, Perceptron & p )
{
	in >> static_cast< NeuralNet & >( p );

	int	nOutputs;
	in >> nOutputs;

	p.m_aOutputUnits.resize( nOutputs );

	for ( int i = 0; i < nOutputs; i++ )
	{
		in >> p.m_aOutputUnits[i];
	}

	return in;
}
