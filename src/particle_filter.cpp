/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

#define EPS 0.0001

using namespace std;

 void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    this->num_particles = 100;  // TODO: Set the number of particles

    /*random generator for x, y, yaw*/
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0, std[0]);
    std::normal_distribution<double> dist_y(0, std[1]);
    std::normal_distribution<double> dist_theta(0, std[2]);

    for (int i = 1; i < this->num_particles + 1; ++i)
    {
        Particle p;
        p.id = i;
        p.x = x + dist_x(gen);
        p.y = y + dist_y(gen);
        p.theta = theta + dist_theta(gen);
        p.weight = 1.0;
        this->particles.push_back(p);
    }

    this->is_initialized = true;
 }

 void ParticleFilter::prediction(double delta_t, double std_pos[],
     double velocity, double yaw_rate) {
     /**
      * TODO: Add measurements to each particle and add random Gaussian noise.
      * NOTE: When adding noise you may find std::normal_distribution
      *   and std::default_random_engine useful.
      *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
      *  http://www.cplusplus.com/reference/random/default_random_engine/
      */

      /*Add random Gaussian noise*/
     std::default_random_engine gen;
     std::normal_distribution<double> dist_x(0, std_pos[0]);
     std::normal_distribution<double> dist_y(0, std_pos[1]);
     std::normal_distribution<double> dist_theta(0, std_pos[2]);

     /*Be careful, do not forget &!*/
     for (Particle &p : this->particles)
     {
         double x = p.x;
         double y = p.y;
         double theta = p.theta;

         /*Use kinematic formulas to predict x, y, yaw*/
         /*consider about a very small yaw_rate is very import!*/
         if (fabs(yaw_rate) < EPS)
         {
             x = x + velocity * delta_t * cos(theta);
             y = y + velocity * delta_t * sin(theta);
         }
         else
         {
             double c1 = theta + yaw_rate * delta_t;
             x = x + velocity / yaw_rate * (sin(c1) - sin(theta));
             y = y + velocity / yaw_rate * (cos(theta) - cos(c1));
             theta = c1;
         }

         p.x = x + dist_x(gen);
         p.y = y + dist_x(gen);
         p.theta = theta + dist_theta(gen);
     }
 }

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  unsigned int nObservations = observations.size();
  unsigned int nPredictions = predicted.size();

  for (unsigned int i = 0; i < nObservations; i++) { // For each observation

    // Initialize min distance as a really big number.
    double minDistance = numeric_limits<double>::max();

    // Initialize the found map in something not possible.
    int mapId = -1;

    for (unsigned j = 0; j < nPredictions; j++ ) { // For each predition.

      double xDistance = observations[i].x - predicted[j].x;
      double yDistance = observations[i].y - predicted[j].y;

      double distance = xDistance * xDistance + yDistance * yDistance;

      // If the "distance" is less than min, stored the id and update min.
      if ( distance < minDistance ) {
        minDistance = distance;
        mapId = predicted[j].id;
      }
    }

    // Update the observation identifier.
    observations[i].id = mapId;
  }
}

 void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    std::vector<LandmarkObs> observations, Map map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */
    double prob_sum = 0.0;
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double norm = 2 * M_PI * sig_x * sig_y;
    vector<double> pre_probs;

    for (Particle p : this->particles)
    {
        double x = p.x;
        double y = p.y;
        double theta = p.theta;
        double joint_prob = 1.0;
        for (LandmarkObs ob : observations)
        {
            double x_obs = ob.x;
            double y_obs = ob.y;

            // 1.转换坐标
            double x_map = x + (cos(theta) * x_obs) - (sin(theta) * y_obs);
            double y_map = y + (sin(theta) * x_obs) + (cos(theta) * y_obs);

            // 2.最近匹配(或者获得最近匹配的x,y)
            double mu_x = 0.0;
            double mu_y = 0.0;
            double min_distance = sensor_range;
            for (Map::single_landmark_s mark : map_landmarks.landmark_list)
            {
                double distance = sqrt(pow(x_map - mark.x_f, 2) + pow(y_map - mark.y_f, 2));
                if (distance > sensor_range)
                    continue;
                if (distance < min_distance)
                {
                    min_distance = distance;
                    mu_x = mark.x_f;
                    mu_y = mark.y_f;
                }
            }

            // 3.通过多元高斯概率密度计算权重（概率）
            double exponent = pow(x_map - mu_x, 2) / pow(sig_x, 2) / 2 
                + pow(y_map - mu_y, 2) / pow(sig_y, 2) / 2;
            double prob = exp(-exponent) / norm;
            joint_prob *= prob;
        }
        pre_probs.push_back(joint_prob);
        prob_sum += joint_prob;
    }
    // 4.归一化权重，并赋值给particles
    for (unsigned int i = 0; i < pre_probs.size(); ++i)
    {
        this->particles[i].weight = pre_probs[i] / prob_sum;
    }
 }

 void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    vector<Particle> new_particles;
    std::default_random_engine gen;
    vector<double> weights;
    for (Particle &p : this->particles)
        weights.push_back(p.weight);
    /*set list of probabilities*/
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    for (int i = 0; i < this->num_particles; ++i)
    {
        int idx = d(gen);
        new_particles.push_back(this->particles[idx]);
    }
    this->particles.clear();
    this->particles = new_particles;
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