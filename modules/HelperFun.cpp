#include <iostream>
#include <math.h>  
#include <vector>  
#include <array>
#include "HelperFun.h"

std::vector<float> HelperFun::LabelTransform(float l) {
	std::vector<float> label(15,0);
        label[l/1000 - 1] = 1;
	return label;
	
	}

float HelperFun::DepositionsTransform(const float i) {return ((i + 1) * (255.))/2;}


std::vector<std::array<double,3>> HelperFun::CellPositionsGeneration() {
	std::vector<std::array<double,3>> pos;
	int dz = 2;
	for (int i = 0; i < 24; i++) {
		for(int j = 0; j < 24; j++) {
			for (int k = 0; k < 24; k++) {
				pos.push_back({(i * cos(j)) * dz - 22, (i * sin(j)) * dz - 22 , dz/2 + k * dz});
			}
		}

	} 

	return pos;
}
