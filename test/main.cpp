/** @file *//********************************************************************************************************

                                                       main.cpp

						                    Copyright 2003, John J. Bolton
	--------------------------------------------------------------------------------------------------------------

	$Header: //depot/Libraries/NeuralNet/Test/main.cpp#3 $

	$NoKeywords: $

 ********************************************************************************************************************/

#include "../Perceptron.h"

#include "Misc/Random.h"
#include "Misc/Etc.h"

#include <cstdio>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

static void TestPerceptron();
static void TestMFF();

Random	rnd( 1 );


/********************************************************************************************************************/
/*																													*/
/*																													*/
/********************************************************************************************************************/

void main( int argc, char ** argv )
{
	TestPerceptron();

	TestMFF();
}


/********************************************************************************************************************/
/*																													*/
/*																													*/
/********************************************************************************************************************/

static void TestPerceptron()
{
	float	c_paWeights[(3+1)*4] =
	{
		 1,  2,  3, 0,
		 4,  5,  6, 0,
		 7,  8,  9, 0,
		10, 11, 12, 0
	};

	Perceptron	a;
	Perceptron	c( 3, 4, Neuron::WeightVector( Neuron::WeightVector::const_iterator( c_paWeights ),
											   Neuron::WeightVector::const_iterator( c_paWeights ) + elementsof( c_paWeights ) ) );

	int const NUM_INPUTS	= 11;

	Neuron::InputVector	b_aInputs( NUM_INPUTS+1 );

	Perceptron	b( NUM_INPUTS+1, 1 );

	for ( int i = 0; i < 10000; i++ )
	{
		std::cout << "i = " << i << ", inputs = ";

		int count = 0;

		for ( int j = 0; j < NUM_INPUTS; j++ )
		{
			bool const	one	= ( ( rnd.Get() & 0x00008000 ) != 0 );

			count += one;

			b_aInputs[j] = float( one );

			std::cout << int(one) << ' ';
		}
		b_aInputs[NUM_INPUTS] = -1.f;
		std::cout << -1 << std::endl;

		Perceptron::OutputVector o	= b( b_aInputs );
		assert( o.size() == 1 );

		Perceptron::ErrorVector	error( 1, float( count >= (NUM_INPUTS+1)/2 ) - o[0] ); 

		std::cout << "T = " << int( count >= (NUM_INPUTS+1)/2 )
				  << ", O = " << o[0]
				  << ", error = " << error[0]
				  << std::endl << std::endl;

		b.Train( b_aInputs, error, 100.f );
	}
}


/********************************************************************************************************************/
/*																													*/
/*																													*/
/********************************************************************************************************************/

static void TestMFF()
{
	enum ConditionIds
	{
		WAITESTIMATE,		// 0 to 1, wait estimate
		TYPE,				// 0, 1, 2, or 3, type of restaurant
		HUNGRY,				// 0 or 1, 1 if we are hungry
		ALTERNATE,			// 0 or 1, 1 if there is a an alternative nearby
		BAR,				// 0 or 1, 1 if there is a bar to wait in
		RAINING,			// 0 or 1, 1 if it is raining
		FRISAT,				// 0 or 1, 1 if it is Friday or Saturday
		PATRONS,			// 0 to 1, how full the restaurant is
		PRICE,				// 0 to 1, how expensive it is
		RESERVATION,		// 0 or 1, 1 if we have made a reservation

		NUM_CONDITIONS
	};

	Neuron::InputVector	Conditions( NUM_CONDITIONS );

}
