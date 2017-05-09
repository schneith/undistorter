#include <string>
#include <boost/bind.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <undistorter/undistorter.h>

const std::string WINDOW_NAME("undistorted view");

void imageCallback(const sensor_msgs::ImageConstPtr& msg, const undistorter::PinholeUndistorter& undistorter)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  //undistort the image
  cv::Mat undist_image;
  undistorter.undistortImage(cv_ptr->image, undist_image);

  cv::imshow(WINDOW_NAME, undist_image);
  cv::waitKey(3);
}

int main(int argc, char **argv)
{
  using namespace undistorter;

  /////////////////////////
  // setup the camera
  /////////////////////////
  Eigen::Vector2d focalLength(461.487, 460.111);
  Eigen::Vector2d principalPoint(356.391, 231.157);
  Eigen::Vector2i resolution(752, 480);

#define USE_EQUIDISTANT

#ifdef USE_EQUIDISTANT
  Eigen::Vector4d distCoeffs_Equi(-0.001650, 0.024372, -0.035828, 0.019860);
  PinholeGeometry camera(focalLength, principalPoint, resolution, EquidistantDistortion::create(distCoeffs_Equi));
#else
  Eigen::Vector4d distCoeffs_RadTan(-0.277226, 0.07025890, 0.00029349, 0.000100);
  PinholeGeometry camera(focalLength, principalPoint, resolution, RadialTangentialDistortion::create(distCoeffs_RadTan));
#endif

  ////////////////////////////
  // create the undistorter
  ////////////////////////////
  double alpha = 1.0, //alphe=0.0: all pixels valid, alpha=1.0: no pixels lost
         scale = 1.0;
  int interpolation = cv::INTER_LINEAR;
  PinholeUndistorter undistorter(camera, alpha, scale, interpolation);

  ////////////////////////////
  // ROS stuffs
  ////////////////////////////
  ros::init(argc, argv, "example_undistorter");

  ros::NodeHandle nh;

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/cam0/image_raw", 1, boost::bind(imageCallback, _1, undistorter));

  ros::spin();

  return 0;
}
