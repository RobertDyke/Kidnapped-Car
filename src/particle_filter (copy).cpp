

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

using std::default_random_engine;
using std::normal_distribution;

const int starting_num_particles = 100;
const double init_weight = 1.0;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//This function comes from lecture 14.4.
    //double std_x, std_y, std_theta; In lecture but not needed here as std[] replaces these three.
	num_particles = starting_num_particles;
    //particles.resize(num_particles);
    //weights.resize(num_particles);

    default_random_engine gen;
    normal_distribution<double> dist_x(x,std[0]);
    normal_distribution<double> dist_y(y,std[1]);
    normal_distribution<double> dist_theta(theta,std[2]);

    for(int i=0;i<num_particles;i++){
    	Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight};
    	//p[i].id = i;
    	//p[i].x = dist_x(gen);
    	//p[i].y = dist_y(gen);
    	//p[i].theta = dist_theta(gen);
    	//p[i].weight = init_weight;
    	particles.push_back(p);
    	weights.push_back(init_weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	normal_distribution<double> dist_x(0,std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_theta(0,std_pos[2]);

	for(unsigned i = 0; i < num_particles;i++){
		if (fabs(yaw_rate) <=0.0){
			particles[i].x = particles[i].x  + velocity * cos(particles[i].theta)* delta_t;
			particles[i].y = particles[i].y  + velocity * sin(particles[i].theta)* delta_t ;
			std::cout<<"prediction A\n";
		}
		else{
			//std::cout<<"prediction B\n";
			particles[i].x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) -
					sin(particles[i].theta));
		    particles[i].y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta +
		    		yaw_rate * delta_t));
		    particles[i].theta = particles[i].theta + yaw_rate * delta_t;
		}

		//adding noise
		particles[i].x  = particles[i].x + dist_x(gen);
		particles[i].y  = particles[i].y + dist_y(gen);
		particles[i].theta  = particles[i].theta + dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned i = 0;i < observations.size(); i++){
		//double min_dist = 0.0;
		double min_dist = numeric_limits<double>::max();
		int map_id = -1;
		LandmarkObs obs = observations[i];

		for(unsigned j = 0; j < predicted.size(); j++){
			LandmarkObs pred = predicted[j];
			double current_dist = dist(obs.x, obs.y, pred.x, pred.y);

			 if(current_dist < min_dist){
				 min_dist = current_dist;
				 map_id = pred.id;
			 }
		}
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//std::cout<<"updateWeights\n";

	for (unsigned i=0; i<num_particles;i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		//transform car coordinates to map coordinates
		vector<LandmarkObs> observations_on_map;
		for(unsigned j = 0;j<observations.size();j++){
			int o_id = observations[j].id;
			double o_x = observations[j].x;
			double o_y = observations[j].y;

			double x_new = p_x + o_x*cos(p_theta) - o_y*sin(p_theta);
			double y_new = p_y + o_x*sin(p_theta) + o_y*cos(p_theta);

			LandmarkObs o_new = {o_id,x_new,y_new};
			observations_on_map.push_back(o_new);
		}
         // find nearby landmarks

		vector<LandmarkObs> landmarks_in_range;
		for(unsigned j=0;j<map_landmarks.landmark_list.size();j++){
			int lm_id = map_landmarks.landmark_list[j].id_i;
			double lm_x = map_landmarks.landmark_list[j].x_f;
			double lm_y = map_landmarks.landmark_list[j].y_f;

			if(dist(p_x, p_y, lm_x, lm_y)<sensor_range){
				landmarks_in_range.push_back(LandmarkObs{lm_id,lm_x,lm_y});
			}
		}
		 // Find landmark that is probably being seen based on Nearest Neighbor algorithm.
		dataAssociation(landmarks_in_range, observations_on_map);

		// Generate weights by taking the difference between particle observations and actual observations.
		particles[i].weight = 1.0;
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		double na = 2.0 * std_x * std_x;
		double nb = 2.0 * std_y * std_y;
		double gauss_norm = 2.0*M_PI*std_x*std_y;

		for(unsigned j=0;j<observations_on_map.size();j++){
			int o_id = observations_on_map[j].id;
			double o_x = observations_on_map[j].x;
			double o_y = observations_on_map[j].y;
			double pr_x;
			double pr_y;
			for(unsigned k=0;k<landmarks_in_range.size();k++){
				if(landmarks_in_range[k].id==o_id){
					pr_x = landmarks_in_range[k].x;
					pr_y = landmarks_in_range[k].y;
					break;
				}
			}
			double obs_w = 1/gauss_norm * exp(-(pow(pr_x-o_x,2)/na + (pow(pr_y-o_y,2)/nb)));
			//multiply observed weight with total weights
			particles[i].weight = particles[i].weight * obs_w;
		}
		weights[i] = particles[i].weight;


	}
}








void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	//std::cout<<"resample\n";
	default_random_engine gen;
	vector<Particle> new_particles;
	vector<double> weights;
	//std::cout<<"num_particles = "<<num_particles<<"\n";
	for(int i=0; i<num_particles;i++){
		weights.push_back(particles[i].weight);
		//std::cout<<"B resample\n";
	}
	discrete_distribution<int> index(weights.begin(),weights.end());
	for(unsigned j=0;j<num_particles;j++){
		//std::cout<<"C resample\n";
		const int i = index(gen);
		new_particles.push_back(particles[i]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
