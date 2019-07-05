/** @file *//********************************************************************************************************

                                              MultilayerFeedForward.cpp

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/MultilayerFeedForward.cpp#3 $

	$NoKeywords: $

 ********************************************************************************************************************/

#include "MultilayerFeedForward.h"

#include <vector>
#include <iostream>
#include <cassert>


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

MultilayerFeedForward::MultilayerFeedForward()
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	nInputs		The number of inputs.
//! @param	nHidden		The number of hidden units.
//! @param	nOutputs	The number of outputs.

MultilayerFeedForward::MultilayerFeedForward( int nInputs, int nHidden, int nOutputs )
	: NeuralNet( nInputs, nOutputs ),
	m_aHiddenUnits( nHidden, Neuron( nInputs ) ),
	m_aHiddenOutputs( nHidden ),
	m_aHiddenGradients( nHidden ),
	m_aOutputUnits( nOutputs, Neuron( nHidden ) ),
	m_aOutputGradients( nOutputs )
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	nInputs		The number of inputs.
//! @param	nHidden		The number of hidden units.
//! @param	nOutputs	The number of outputs.
//! @param	aWeights	The weights for each input of each unit. There must be
//!						( @a nInputs + @a nOutputs ) * @a nHidden weights. The first @a nInputs * @a nHidden
//!						weights are the input weights for the hidden units. The next @a nOutputs * @a nHidden
//!						weights (the rest) are the input weights for the output units.

MultilayerFeedForward::MultilayerFeedForward( int nInputs, int nHidden, int nOutputs,
											  Neuron::WeightVector const & aWeights )
	: NeuralNet( nInputs, nOutputs ),
	m_aHiddenUnits( nHidden ),
	m_aHiddenOutputs( nHidden ),
	m_aHiddenGradients( nHidden ),
	m_aOutputUnits( nOutputs ),
	m_aOutputGradients( nOutputs )
{
	assert( (int)aWeights.size() == ( nInputs + nOutputs ) * nHidden );

	Neuron::WeightVector::const_iterator	pFirst	= aWeights.begin();

	// Initialize the input weights for the hidden units

	for ( int j = 0; j < nHidden; j++ )
	{
		m_aHiddenUnits[j].Initialize( Neuron::WeightVector( pFirst, pFirst + nInputs ) );
		pFirst += nInputs;
	}

	// Initialize the input weights for the output units

	for ( int i = 0; i < nOutputs; i++ )
	{
		m_aOutputUnits[i].Initialize( Neuron::WeightVector( pFirst, pFirst + nHidden ) );
		pFirst += nHidden;
	}
}

/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

MultilayerFeedForward::~MultilayerFeedForward()
{
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

MultilayerFeedForward::OutputVector const & MultilayerFeedForward::operator()( Neuron::InputVector const & aInputs )
{
	// Update the hidden outputs.

	int const	nHidden	= (int)m_aHiddenUnits.size();

	for ( int j = 0; j < nHidden; j++ )
	{
		m_aHiddenOutputs[j] = m_aHiddenUnits[j]( aInputs, &m_aHiddenGradients[j] );
	}

	// Update the outputs.

	int const	nOutputs	= (int)m_aOutputUnits.size();

	for ( int i = 0; i < nOutputs; i++ )
	{
		m_aOutputs[i] = m_aOutputUnits[i]( m_aHiddenOutputs, &m_aOutputGradients[i] );
	}

	return m_aOutputs;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

void MultilayerFeedForward::Train( Neuron::InputVector const & aInputs, ErrorVector const & aErrors, float rate )
{
	assert( aInputs.size() == m_nInputs );
	assert( aErrors.size() == m_aOutputUnits.size() );

	int const	nOutputs	= (int)m_aOutputUnits.size();

	for ( int i = 0; i < nOutputs; i++ )
	{
		m_aOutputGradients[i] *= aErrors[i];
		m_aOutputUnits[i].AdjustWeights( m_aHiddenOutputs, m_aOutputGradients[i], rate );
	}

	int const	nHidden	= (int)m_aHiddenUnits.size();

	for ( int j = 0; j < nHidden; j++ )
	{
		float	s	= 0.f;
		for ( int i = 0; i < nOutputs; i++ )
		{
			s += m_aOutputUnits[i].GetWeights()[j] * m_aOutputGradients[i];
		}
		m_aHiddenGradients[j] *= s;

		m_aHiddenUnits[j].AdjustWeights( aInputs, m_aHiddenGradients[j], rate );
	}

}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	out		The output stream.
//! @param	mff		The MultilayerFeedForward to output.

std::ostream & operator<<( std::ostream & out, MultilayerFeedForward const & mff )
{
	out << static_cast< NeuralNet const & >( mff );

	int const	nHidden	= (int)mff.m_aHiddenUnits.size();
	int const	nOutputs	= (int)mff.m_aOutputUnits.size();

	out << ' ' << nHidden << ' ' << nOutputs << std::endl;

	for ( int i = 0; i < nHidden; i++ )
	{
		out << mff.m_aHiddenUnits[i] << std::endl;
	}

	for ( int j = 0; j < nOutputs; j++ )
	{
		out << mff.m_aOutputUnits[j] << std::endl;
	}

	return out;
}


/********************************************************************************************************************/
/*																													*/
/********************************************************************************************************************/

//! @param	in		The input stream.
//! @param	mff		The MultilayerFeedForward to input.

std::istream & operator>>( std::istream & in, MultilayerFeedForward & mff )
{
	in >> static_cast< NeuralNet & >( mff );

	int		nOutputs;
	int		nHidden;

	in >> nHidden >> nOutputs;

	mff.m_aHiddenUnits.resize( nHidden );
	mff.m_aOutputUnits.resize( nOutputs );

	for ( int i = 0; i < nHidden; i++ )
	{
		in >> mff.m_aHiddenUnits[i];
	}

	for ( int j = 0; j < nOutputs; j++ )
	{
		in >> mff.m_aOutputUnits[j];
	}

	return in;
}
