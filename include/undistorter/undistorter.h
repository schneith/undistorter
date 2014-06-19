//Thomas Schneider, 2014
//based on aslam distortion code

#ifndef UNDISTORTER_HPP_
#define UNDISTORTER_HPP_

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>


namespace undistorter {

//////////////////////////////////////////////////////
// Distortion models
//////////////////////////////////////////////////////
class DistortionModel {
 public:
  virtual void distort(Eigen::Vector2d& y) const = 0;
  virtual void undistort(Eigen::Vector2d& y) const = 0;

  typedef boost::shared_ptr<DistortionModel> Ptr;
  typedef boost::shared_ptr<const DistortionModel> ConstPtr;
};

class NullDistortion : public DistortionModel
{
 public:
  typedef boost::shared_ptr<NullDistortion> Ptr;
  typedef boost::shared_ptr<const NullDistortion> ConstPtr;

  static Ptr create()
  {
    return boost::make_shared<NullDistortion>();
  }

  NullDistortion() {};
  virtual ~NullDistortion() {};

  void distort(Eigen::Vector2d& y) const {};
  void undistort(Eigen::Vector2d& y) const {};
};


class EquidistantDistortion : public DistortionModel
{
 public:
  typedef boost::shared_ptr<EquidistantDistortion> Ptr;
  typedef boost::shared_ptr<const EquidistantDistortion> ConstPtr;

  static Ptr create(const Eigen::Vector4d& distCoeffs)
  {
    return boost::make_shared<EquidistantDistortion>(distCoeffs);
  }

  EquidistantDistortion(const Eigen::Vector4d& distCoeffs)
   : _distCoeffs(distCoeffs) {};

  virtual ~EquidistantDistortion() {};

  void distort(Eigen::Vector2d& y) const
  {
    double r, theta, theta2, theta4, theta6, theta8, thetad, scaling;

    r = sqrt(y[0]*y[0] + y[1]*y[1]);
    theta = atan(r);
    theta2 = theta * theta;
    theta4 = theta2 * theta2;
    theta6 = theta4 * theta2;
    theta8 = theta4 * theta4;
    thetad = theta * (1 + _distCoeffs[0] * theta2 + _distCoeffs[1] * theta4 + _distCoeffs[2] * theta6 + _distCoeffs[3] * theta8);
    scaling = (r > 1e-8) ? thetad / r : 1.0;
    y[0] *= scaling;
    y[1] *= scaling;
  };

  void undistort(Eigen::Vector2d& y) const
  {
    double theta, theta2, theta4, theta6, theta8, thetad, scaling;

    thetad = sqrt(y[0] * y[0] + y[1] * y[1]);
    theta = thetad;  // initial guess
    for (int i = 20; i > 0; i--) {
      theta2 = theta * theta;
      theta4 = theta2 * theta2;
      theta6 = theta4 * theta2;
      theta8 = theta4 * theta4;
      theta = thetad / (1 + _distCoeffs[0] * theta2 + _distCoeffs[1] * theta4 + _distCoeffs[2] * theta6 + _distCoeffs[3] * theta8);
    }

    scaling = tan(theta) / thetad;
    y[0] *= scaling;
    y[1] *= scaling;
  };

 private:
  const Eigen::Vector4d _distCoeffs;
};


class RadialTangentialDistortion : public DistortionModel
{
 public:
  typedef boost::shared_ptr<RadialTangentialDistortion> Ptr;
  typedef boost::shared_ptr<const RadialTangentialDistortion> ConstPtr;

  static Ptr create(const Eigen::Vector4d& distCoeffs)
  {
    return boost::make_shared<RadialTangentialDistortion>(distCoeffs);
  }

  RadialTangentialDistortion(const Eigen::Vector4d& distCoeffs)
   : _distCoeffs(distCoeffs) {};

  virtual ~RadialTangentialDistortion() {};

  void distort(Eigen::Vector2d& y) const
  {
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = y[0] * y[0];
    my2_u = y[1] * y[1];
    mxy_u = y[0] * y[1];
    rho2_u = mx2_u + my2_u;
    rad_dist_u = _distCoeffs[0] * rho2_u + _distCoeffs[1] * rho2_u * rho2_u;
    y[0] += y[0] * rad_dist_u + 2.0 * _distCoeffs[2] * mxy_u + _distCoeffs[3] * (rho2_u + 2.0 * mx2_u);
    y[1] += y[1] * rad_dist_u + 2.0 * _distCoeffs[3] * mxy_u + _distCoeffs[2] * (rho2_u + 2.0 * my2_u);
  };

  void undistort(Eigen::Vector2d& y) const
  {
    const int n = 5;
    Eigen::Vector2d ybar = y;
    Eigen::Matrix2d J;

    Eigen::Vector2d y_tmp;

    for (int i = 0; i < n; i++)
    {
      y_tmp = ybar;

      double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
      mx2_u = y_tmp[0] * y_tmp[0];
      my2_u = y_tmp[1] * y_tmp[1];
      mxy_u = y_tmp[0] * y_tmp[1];
      rho2_u = mx2_u + my2_u;
      rad_dist_u = _distCoeffs[0] * rho2_u + _distCoeffs[1] * rho2_u * rho2_u;

      J(0, 0) = 1 + rad_dist_u + _distCoeffs[0] * 2.0 * mx2_u + _distCoeffs[1] * rho2_u * 4 * mx2_u
          + 2.0 * _distCoeffs[2] * y_tmp[1] + 6 * _distCoeffs[3] * y_tmp[0];
      J(1, 0) = _distCoeffs[0] * 2.0 * y_tmp[0] * y_tmp[1] + _distCoeffs[1] * 4 * rho2_u * y_tmp[0] * y_tmp[1]
          + _distCoeffs[2] * 2.0 * y_tmp[0] + 2.0 * _distCoeffs[3] * y_tmp[1];
      J(0, 1) = J(1, 0);
      J(1, 1) = 1 + rad_dist_u + _distCoeffs[0] * 2.0 * my2_u + _distCoeffs[1] * rho2_u * 4 * my2_u
          + 6 * _distCoeffs[2] * y[1] + 2.0 * _distCoeffs[3] * y[0];
      y_tmp[0] += y_tmp[0] * rad_dist_u + 2.0 * _distCoeffs[2] * mxy_u + _distCoeffs[3] * (rho2_u + 2.0 * mx2_u);
      y_tmp[1] += y_tmp[1] * rad_dist_u + 2.0 * _distCoeffs[3] * mxy_u + _distCoeffs[2] * (rho2_u + 2.0 * my2_u);

      Eigen::Vector2d e(y - y_tmp);
      Eigen::Vector2d du = (J.transpose() * J).inverse() * J.transpose() * e;
      ybar += du;

      if (e.dot(e) < 1e-15)
        break;
    }
    y = ybar;
  };

 private:
  //_distCoeffs: k1, k2, p1, p2
  const Eigen::Vector4d _distCoeffs;
};


//////////////////////////////////////////////////////
// Camera Model
//////////////////////////////////////////////////////
struct PinholeGeometry {
  PinholeGeometry()
      : focallength(Eigen::Vector2d(0,0)),
        principalpoint(Eigen::Vector2d(0,0)),
        resolution(Eigen::Vector2i(0,0)),
        distortion(NullDistortion::create())
  {};

  PinholeGeometry(const Eigen::Vector2d& focallength,
                  const Eigen::Vector2d& principalpoint,
                  const Eigen::Vector2i& resolution,
                  DistortionModel::Ptr distortion)
      : focallength(focallength),
        principalpoint(principalpoint),
        resolution(resolution),
        distortion(distortion)
  {};

  PinholeGeometry(const Eigen::Matrix3d& cameraMatrix,
                  const Eigen::Vector2i& resolution,
                  DistortionModel::Ptr distortion)
      : resolution(resolution),
        distortion(distortion)
  {
    focallength[0] = cameraMatrix(0, 0);
    focallength[1] = cameraMatrix(1, 1);
    principalpoint[0] = cameraMatrix(0, 2);
    principalpoint[1] = cameraMatrix(1, 2);
  };

  Eigen::Matrix3d getCameraMatrix() const
  {
    Eigen::Matrix3d cameraMatrix = Eigen::Matrix3d::Zero();
    cameraMatrix(0,0) = focallength[0];
    cameraMatrix(1,1) = focallength[1];
    cameraMatrix(0,2) = principalpoint[0];
    cameraMatrix(1,2) = principalpoint[1];
    cameraMatrix(2,2) = 1.0;
    return cameraMatrix;
  }

  //camera parameters
  Eigen::Vector2d focallength;
  Eigen::Vector2d principalpoint;
  Eigen::Vector2i resolution;

  //distortion
  DistortionModel::Ptr distortion;
};

//////////////////////////////////////////////////////
// Custom OpenCV functions
//////////////////////////////////////////////////////
namespace cv_helper {
using namespace cv;
///
//calculates the inner(min)/outer(max) rectangle on the undistorted image
//
// INPUT:
//  camera_geometry:  camera geometry (distortion and intrinsics used)
//  imgSize:          The original image size.
// OUTPUT:
//  inner:            inner rectangle (all pixels valid)
//  outer:            outer rectangle (no pixels lost)
///
static void icvGetRectangles(const PinholeGeometry& camera_geometry,
                             CvSize imgSize, cv::Rect_<float>& inner,
                             cv::Rect_<float>& outer) {
  const int N = 9;
  int x, y, k;
  cv::Ptr<CvMat> _pts(cvCreateMat(1, N * N, CV_32FC2));
  CvPoint2D32f* pts = (CvPoint2D32f*) (_pts->data.ptr);

  for (y = k = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      Eigen::Vector2d point(x * imgSize.width / (N - 1),
                            y * imgSize.height / (N - 1));

      //normalize
      Eigen::Matrix3d cameraMatrix = camera_geometry.getCameraMatrix();
      double cu = cameraMatrix(0, 2), cv = cameraMatrix(1, 2);
      double fu = cameraMatrix(0, 0), fv = cameraMatrix(1, 1);
      point(0) = (point(0) - cu) / fu;
      point(1) = (point(1) - cv) / fv;

      //undistort
      camera_geometry.distortion->undistort(point);

      pts[k++] = cvPoint2D32f((float) point[0], (float) point[1]);
    }
  }

  float iX0 = -FLT_MAX, iX1 = FLT_MAX, iY0 = -FLT_MAX, iY1 = FLT_MAX;
  float oX0 = FLT_MAX, oX1 = -FLT_MAX, oY0 = FLT_MAX, oY1 = -FLT_MAX;
  // find the inscribed rectangle.
  // the code will likely not work with extreme rotation matrices (R) (>45%)
  for (y = k = 0; y < N; y++)
    for (x = 0; x < N; x++) {
      CvPoint2D32f p = pts[k++];
      oX0 = MIN(oX0, p.x);
      oX1 = MAX(oX1, p.x);
      oY0 = MIN(oY0, p.y);
      oY1 = MAX(oY1, p.y);

      if (x == 0)
        iX0 = MAX(iX0, p.x);
      if (x == N - 1)
        iX1 = MIN(iX1, p.x);
      if (y == 0)
        iY0 = MAX(iY0, p.y);
      if (y == N - 1)
        iY1 = MIN(iY1, p.y);
    }
  inner = cv::Rect_<float>(iX0, iY0, iX1 - iX0, iY1 - iY0);
  outer = cv::Rect_<float>(oX0, oY0, oX1 - oX0, oY1 - oY0);
}

///
//Returns the new camera matrix based on the free scaling parameter.
//
// INPUT:
//  camera_geometry:  camera geometry (distortion and intrinsics used)
//  imgSize:          The original image size.
//  alpha:            Free scaling parameter between 0 (when all the pixels in the undistorted image will be valid)
//                    and 1 (when all the source image pixels will be retained in the undistorted image)
//  newImgSize:       Image size after rectification. By default it will be set to imageSize.
// RETURNS:           The output new camera matrix.
///
Eigen::Matrix3d getOptimalNewCameraMatrix(
    const PinholeGeometry& camera_geometry, CvSize imgSize, double alpha,
    CvSize newImgSize) {

  cv::Rect_<float> inner, outer;
  newImgSize = newImgSize.width * newImgSize.height != 0 ? newImgSize : imgSize;

  // Get inscribed and circumscribed rectangles in normalized
  // (independent of camera matrix) coordinates
  icvGetRectangles(camera_geometry, imgSize, inner, outer);

  // Projection mapping inner rectangle to viewport
  double fx0 = (newImgSize.width - 1) / inner.width;
  double fy0 = (newImgSize.height - 1) / inner.height;
  double cx0 = -fx0 * inner.x;
  double cy0 = -fy0 * inner.y;

  // Projection mapping outer rectangle to viewport
  double fx1 = (newImgSize.width - 1) / outer.width;
  double fy1 = (newImgSize.height - 1) / outer.height;
  double cx1 = -fx1 * outer.x;
  double cy1 = -fy1 * outer.y;

  // Interpolate between the two optimal projections
  Eigen::Matrix3d newCameraMatrixEigen = Eigen::Matrix3d::Zero();
  newCameraMatrixEigen(0, 0) = fx0 * (1 - alpha) + fx1 * alpha;
  newCameraMatrixEigen(1, 1) = fy0 * (1 - alpha) + fy1 * alpha;
  newCameraMatrixEigen(0, 2) = cx0 * (1 - alpha) + cx1 * alpha;
  newCameraMatrixEigen(1, 2) = cy0 * (1 - alpha) + cy1 * alpha;
  newCameraMatrixEigen(2, 2) = 1.0;

  return newCameraMatrixEigen;
}

///
//Returns the new camera matrix based on the free scaling parameter.
//
// INPUT:
//  camera_geometry:  camera geometry (distortion and intrinsics used)
//  R:                Optional rectification transformation in the object space (3x3 matrix)
//  newCameraMatrix:  New camera matrix
//  size:             Undistorted image size.
//  m1type:           Type of the first output map that can be CV_32FC1 or CV_16SC2
//
// OUTPUT: (maps can be used with cv::remap)
//  _map1:            The first output map.
//  _map2:            The second output map.
///
void initUndistortRectifyMap(const PinholeGeometry& _distortedCamera,
                             const Eigen::Matrix3d& _R,
                             const Eigen::Matrix3d& _newCameraMatrix, Size size,
                             int m1type, OutputArray _map1, OutputArray _map2) {

  //prepare the outputs data structures
  if (m1type <= 0)
    m1type = CV_16SC2;
  CV_Assert(m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2);
  _map1.create(size, m1type);
  Mat map1 = _map1.getMat(), map2;
  if (m1type != CV_32FC2) {
    _map2.create(size, m1type == CV_16SC2 ? CV_16UC1 : CV_32FC1);
    map2 = _map2.getMat();
  } else
    _map2.release();

  //invert
  Eigen::Matrix3d invR = (_newCameraMatrix * _R).inverse();

  for (int i = 0; i < size.height; i++) {
    float* m1f = (float*) (map1.data + map1.step * i);
    float* m2f = (float*) (map2.data + map2.step * i);
    short* m1 = (short*) m1f;
    ushort* m2 = (ushort*) m2f;

    double _x = i * invR(0, 1) + invR(0, 2),  //TODO maybe the is ordering wrong...
    _y = i * invR(1, 1) + invR(1, 2), _w = i * invR(2, 1) + invR(2, 2);

    for (int j = 0; j < size.width;
        j++, _x += invR(0, 0), _y += invR(1, 0), _w += invR(2, 0)) {
      double w = 1. / _w, x = _x * w, y = _y * w;

      //apply original distortion
      Eigen::Vector2d point(x, y);
      _distortedCamera.distortion->distort(point);
      double u_norm = point[0], v_norm = point[1];

      //apply original camera matrix
      //get the camera matrix from the camera
      Eigen::Matrix3d cameraMatrix = _distortedCamera.getCameraMatrix();

      double u0 = cameraMatrix(0, 2), v0 = cameraMatrix(1, 2);
      double fx = cameraMatrix(0, 0), fy = cameraMatrix(1, 1);

      double u = fx * u_norm + u0;
      double v = fy * v_norm + v0;

      //store in output format
      if (m1type == CV_16SC2) {
        int iu = saturate_cast<int>(u * INTER_TAB_SIZE);
        int iv = saturate_cast<int>(v * INTER_TAB_SIZE);
        m1[j * 2] = (short) (iu >> INTER_BITS);
        m1[j * 2 + 1] = (short) (iv >> INTER_BITS);
        m2[j] = (ushort) ((iv & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE
            + (iu & (INTER_TAB_SIZE - 1)));
      } else if (m1type == CV_32FC1) {
        m1f[j] = (float) u;
        m2f[j] = (float) v;
      } else {
        m1f[j * 2] = (float) u;
        m1f[j * 2 + 1] = (float) v;
      }
    }
  }
}

}

//////////////////////////////////////////////////////
// Undistorter
//////////////////////////////////////////////////////

class PinholeUndistorter
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \param[in] alpha                The free scaling parameter between 0 (when all the pixels in the undistorted image will be valid) and 1 (when all the source image pixels will be retained in the undistorted image).
  /// \param[in] scale                Allows to scale the resulting image.
  PinholeUndistorter(const PinholeGeometry& distortedGeometry, double alpha,
                     double scale, int interpolation)
      : distortedGeometry_(distortedGeometry),
        interpolation_(interpolation)
  {
    int width = distortedGeometry_.resolution[0];
    int height = distortedGeometry_.resolution[1];
    int newWidth = (int) width * scale;
    int newHeight = (int) height * scale;

    // compute the optimal new camera matrix based on the free scaling parameter
    Eigen::Matrix3d idealCameraMatrix = Eigen::Matrix3d::Zero();
    idealCameraMatrix = cv_helper::getOptimalNewCameraMatrix(distortedGeometry_, cv::Size(width, height), alpha, cv::Size(newWidth, newHeight));

    // set new idealGeometry
    idealGeometry_ = PinholeGeometry(idealCameraMatrix,
                                     Eigen::Vector2i(newWidth, newHeight),
                                     boost::make_shared<NullDistortion>());

    // compute the undistortion and rectification transformation map
    cv_helper::initUndistortRectifyMap(distortedGeometry_,
                                       Eigen::Matrix3d::Identity(),
                                       idealCameraMatrix,
                                       cv::Size(newWidth, newHeight), CV_16SC2,
                                       mapX_, mapY_);
  }

  ~PinholeUndistorter() {};

  /// \brief Undistort a single image.
  void undistortImage(const cv::Mat& inImage, cv::Mat& outImage) const
  {
    cv::remap(inImage, outImage, mapX_, mapY_, interpolation_);
  }

 private:
  const PinholeGeometry distortedGeometry_;
  PinholeGeometry idealGeometry_;

  //undistorter map
  cv::Mat mapX_, mapY_;

  //interpolation type
  int interpolation_;
};

} //namespace undistorter

#endif /* UNDISTORTER_HPP_ */

