/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test.h"
#include <limits>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iomanip>
#include <cmath>


//#include "opencv2/objdetect/objdetect_tegra.hpp"

namespace cv
{
namespace my_linemod
{

static bool use63 = true;
static int featuresUsed = 63;
static bool punteggio16 = false;
static bool featuresSignatureCandidates = true;
static bool grayEnabled = true;
static bool onlyOne = true;
static bool onlyOne_1 = true;
static bool onlyOne_2 = true;
static bool DEBUGGING = false;
static bool signatureEnabled = true;
static int countStatic = 0;
static string increment = "big";
//più bassa, meno i colori si accorpano
static int threshold_rgb = 45;
//static float threshold_mag = 50;//percentuale
static int scoreUsed = 0;

static int signFeat = 30;

#define WHITE false
#define BLACK true

singleton_test* singleton_test::instance_ptr = NULL;


void setSingleton()
{
    singleton_test * ptr_singleton = singleton_test::get_instance();    
    if(ptr_singleton->initialized == true)
    {
	cout<<endl<<"-->Singleton caricato<--"<<endl;

	
	threshold_rgb = ptr_singleton->s_threshold_rgb;
	use63 = ptr_singleton->s_use63;
	featuresUsed = ptr_singleton->s_featuresUsed;
	signFeat = ptr_singleton->s_signFeat;
	featuresSignatureCandidates = ptr_singleton->s_featuresSignatureCandidates;
	grayEnabled = ptr_singleton->s_grayEnabled;
	punteggio16 = ptr_singleton->s_punteggio16;
	signatureEnabled = ptr_singleton->s_signatureEnabled;
	
	DEBUGGING = ptr_singleton->s_DEBUGGING;
	
	cout<<"threshold_rgb: "<<threshold_rgb <<endl;
	cout<<"use63: "<< use63<<endl;
	cout<<"featuresUsed: "<< featuresUsed<<endl;
	cout<<"signFeat: "<< signFeat<<endl;
	cout<<"featuresSignatureCandidates: "<<featuresSignatureCandidates <<endl;
	cout<<"grayEnabled: "<< grayEnabled<<endl;
	cout<<"punteggio16: "<< punteggio16<<endl;
	cout<<"signatureEnabled: "<<signatureEnabled <<endl;
	cout<<"DEBUGGING: "<<DEBUGGING <<endl;
    }
    
}


class Timer
{
public:
  Timer() : start_(0), time_(0) {}

  void start()
  {
    start_ = cv::getTickCount();
  }

  void stop()
  {
    CV_Assert(start_ != 0);
    int64 end = cv::getTickCount();
    time_ += end - start_;
    start_ = 0;
  }

  double time()
  {
    double ret = time_ / cv::getTickFrequency();
    time_ = 0;
    return ret;
  }

private:
  int64 start_, time_;
};


// struct Feature

/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
  switch (quantized)
  {
    case 0: return 8;
    case 1:   return 0;
    case 2:   return 1;
    case 4:   return 2;
    case 8:   return 3;
    case 16:  return 4;
    case 32:  return 5;
    case 64:  return 6;
    case 128: return 7;
    //case 129:  return 8; //gray-white
    //case 130:  return 9; //gray-black
    default:
      CV_Assert(false);
      return -1; //avoid warning
  }
}

static inline string getColorFromLabel(int quantized)
{
  switch (quantized)
  {
    case 1:   return "R";
    case 2:   return "G";
    case 4:   return "B";
    case 8:   return "RG";
    case 16:  return "RB";
    case 32:  return "GB";
    case 64:  return "white";
    case 128: return "black";
    default:
      CV_Assert(false);
      return "no color o.O"; //avoid warning
  }
}


void Feature::read(const FileNode& fn)
{
  FileNodeIterator fni = fn.begin();
  fni >> x >> y >> label >> rgbLabel >> onBorder;
}



void Feature::write(FileStorage& fs) const
{
  fs << "[:" << x << y << label << rgbLabel << onBorder << "]";
}

// struct Template

/**
 * \brief Crop a set of overlapping templates from different modalities.
 *
 * \param[in,out] templates Set of templates representing the same object view.
 *
 * \return The bounding box of all the templates in original image coordinates.
 */
Rect cropTemplates(std::vector<Template>& templates, const Mat& maskTemplate)
{

  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();

  // First pass: find min/max feature x,y over all pyramid levels and modalities
  for (int i = 0; i < (int)templates.size(); ++i)
  {
    Template& templ = templates[i];

    for (int j = 0; j < (int)templ.features.size(); ++j)
    {
      int x = templ.features[j].x << templ.pyramid_level;
      int y = templ.features[j].y << templ.pyramid_level;
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);
    }
  }

  /// @todo Why require even min_x, min_y?
  if (min_x % 2 == 1) --min_x;
  if (min_y % 2 == 1) --min_y;

  // Second pass: set width/height and shift all feature positions
  for (int i = 0; i < (int)templates.size(); ++i)
  {
    Template& templ = templates[i];
    templ.width = (max_x - min_x) >> templ.pyramid_level;
    templ.height = (max_y - min_y) >> templ.pyramid_level;

    
	Mat cropMask = maskTemplate(Rect(min_x, min_y, max_x - min_x, max_y - min_y));
	//templ.croppedMask = cropMask;
	//templ.offsetX = min_x;
	//templ.offsetY = min_y;
    
    int offset_x = min_x >> templ.pyramid_level;
    int offset_y = min_y >> templ.pyramid_level;
    
    //Mat drawing = Mat::zeros( cropMask.rows, cropMask.cols, CV_8UC3);
    
    for (int j = 0; j < (int)templ.features.size(); ++j)
    {
      templ.features[j].x -= offset_x;
      templ.features[j].y -= offset_y;
    }
    

    //RIABILITARE FEATURES SIGNATURE
    for (int j = 0; j < (int)templ.featuresSignature.size(); ++j)
    {
      templ.featuresSignature[j].x -= offset_x;
      templ.featuresSignature[j].y -= offset_y;
    }
    
    if(templ.pyramid_level == 0)
    {
	Mat maskTmp;
	cropMask.copyTo(maskTmp);
  
	vector<vector<Point> > contours;
	Mat gray;
	
	findContours( maskTmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1);
	templ.contour = contours[0];
	
	/*	
	Scalar color = Scalar( 255, 0, 0 );
	drawContours( drawing, contours, 0, color);
	//double res = pointPolygonTest( contours[0], Point(1,1), true);

	

	cv::circle(drawing, Point(1,1), 2, Scalar( 0, 0, 255 ));
	if(DEBUGGING)
	{  imshow("contorni", drawing);
	    waitKey(0);
	}*/
	

	
    }

  
  }
    
  return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

void Template::read(const FileNode& fn)
{
  width = fn["width"];
  height = fn["height"];
  pyramid_level = fn["pyramid_level"];
  totalFeatures = fn["totalFeatures"];
  //fn["croppedMask"] >> croppedMask;
  fn["contour"] >> contour;
  FileNode features_fn = fn["features"];
  features.resize(features_fn.size());
  FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
  for (int i = 0; it != it_end; ++it, ++i)
  {
    features[i].read(*it);
  }
  FileNode features_signature_fn = fn["featuresSignature"];
  featuresSignature.resize(features_signature_fn.size());
  
  FileNodeIterator its = features_signature_fn.begin(), its_end = features_signature_fn.end();
  for (int i = 0; its != its_end; ++its, ++i)
  {
    featuresSignature[i].read(*its);
  }
}

void Template::write(FileStorage& fs) const
{
  fs << "width" << width;
  fs << "height" << height;
  fs << "pyramid_level" << pyramid_level;
  fs << "totalFeatures" << totalFeatures;
  //fs << "croppedMask" << croppedMask;
  fs << "contour" << contour;
  fs << "features" << "[";
  for (int i = 0; i < (int)features.size(); ++i)
  {
    features[i].write(fs);
  }
  fs << "]"; // features
  fs << "featuresSignature" << "[";
  for (int i = 0; i < (int)featuresSignature.size(); ++i)
  {
    featuresSignature[i].write(fs);
  }
  fs << "]"; // featuresSignature
}



/****************************************************************************************\
*                             Modality interfaces                                        *
\****************************************************************************************/

void QuantizedPyramid::selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                               std::vector<Feature>& features,
                                               size_t num_features, float distance)
{
  features.clear();
  float distance_sq = CV_SQR(distance);
  int i = 0;
  while (features.size() < num_features)
  {
    Candidate c = candidates[i];

    // Add if sufficient distance away from any previously chosen feature
    bool keep = true;
    for (int j = 0; (j < (int)features.size()) && keep; ++j)
    {
      Feature f = features[j];
      keep = CV_SQR(c.f.x - f.x) + CV_SQR(c.f.y - f.y) >= distance_sq;
    }
    if (keep)
      features.push_back(c.f);

    if (++i == (int)candidates.size())
    {
      // Start back at beginning, and relax required distance
      i = 0;
      distance -= 1.0f;
      distance_sq = CV_SQR(distance);
    }
  }
}

Ptr<Modality> Modality::create(const std::string& modality_type)
{
  if (modality_type == "ColorGradient")
    return new ColorGradient();
  else if (modality_type == "DepthNormal")
    return new DepthNormal();
  else
    return NULL;
}

Ptr<Modality> Modality::create(const FileNode& fn)
{
  std::string type = fn["type"];
  Ptr<Modality> modality = create(type);
  modality->read(fn);
  return modality;
}

void colormap(const Mat& quantized, Mat& dst)
{
  std::vector<Vec3b> lut(8);
  lut[0] = Vec3b(  0,   0, 255);
  lut[1] = Vec3b(  0, 170, 255);
  lut[2] = Vec3b(  0, 255, 170);
  lut[3] = Vec3b(  0, 255,   0);
  lut[4] = Vec3b(170, 255,   0);
  lut[5] = Vec3b(255, 170,   0);
  lut[6] = Vec3b(255,   0,   0);
  lut[7] = Vec3b(255,   0, 170);

  dst = Mat::zeros(quantized.size(), CV_8UC3);
  for (int r = 0; r < dst.rows; ++r)
  {
    const uchar* quant_r = quantized.ptr(r);
    Vec3b* dst_r = dst.ptr<Vec3b>(r);
    for (int c = 0; c < dst.cols; ++c)
    {
      uchar q = quant_r[c];
      if (q)
        dst_r[c] = lut[getLabel(q)];
    }
  }
}

/****************************************************************************************\
*                             Color gradient modality                                    *
\****************************************************************************************/

// Forward declaration
void hysteresisGradient(Mat& magnitude, Mat& angle,
                        Mat& ap_tmp, float w_threshold, float s_threshold, bool computeMagnitudeStrong, Mat& magnitudeStrong);


/** 
 * \brief returns if the values of 3 channels are similar to black or to white 
 * 255+255+255 = 765
 * <382 = white
 * >=382 = black
 */
 
int *bar(int *array)
{
int *start = array;
while ( *array )
{
*array++ += 1;
}
return start;
}

void writeMat(Mat& m)
{
	string fileNameConf = "./matrice_stampata.txt";
	FileStorage fsConf(fileNameConf, FileStorage::WRITE);
    fsConf << "m" << m;

	fsConf.release();
}

void createTableRGB()
{
	//uchar tableRGB[8][8] = {{4,0,0,3,3,0,2,2}, {0,4,0,3,0,3,2,2}, {0,0,4,0,3,3,2,2}, {3,3,0,4,1,1,2,2}, {3,0,3,1,4,1,2,2}, {0,3,3,1,1,4,2,2}, {1,1,1,2,2,2,4,0}, {1,1,1,2,2,2,0,4}};
	uchar tableRGB[8][8] = {{4,0,0,3,3,0,0,0}, {0,4,0,3,0,3,0,0}, {0,0,4,0,3,3,0,0}, {3,3,0,4,1,1,0,0}, {3,0,3,1,4,1,0,0}, {0,3,3,1,1,4,0,0}, {1,1,1,2,2,2,0,0}, {1,1,1,2,2,2,0,0}};
	
}

short isGrayWhite(unsigned char quant, bool grayWhiteOrBlack)
{
    
    if(quant == 0 && grayWhiteOrBlack == WHITE)
	return 64;
	
    if(quant == 0 && grayWhiteOrBlack == BLACK)
	return 128;

    return quant;
}


//0 = black
//1 = white
//2 = gray-black
//3 = gray-white
short isBlack(unsigned short r, unsigned short g, unsigned short b)
{
	ushort sum = r + g + b;
	if(sum <250)
	    return 0;
	else if(sum > 450)
	    return 1;
	else if(sum < 384)
	    return 2; //gray-black
	else 
	    return 3; //gray-white
}

/*bool isBlack(unsigned short r, unsigned short g, unsigned short b)
{
	ushort sum = r + g + b;
	if(sum <384)
	    return true;
	else 
	    return false;
}*/

bool isBlackMag(int r, int g, int b)
{
	int sum = r + g + b;
	if(sum >382)
		return true;
	else
		return false;
}

/** 
 * \brief returns the quantization based on the diffs among values of rgb channels
 * 00000001 = R = 1
 * 00000010 = G = 2
 * 00000100 = B = 4
 * 00001000 = RG = 8
 * 00010000 = RB = 16
 * 00100000 = GB = 32
 * 01000000 = RGB_WHITE = 64
 * 10000000 = RGB_BLACK = 128
 * 
 * \param[in]  r         value of the channel R
 * \param[in]  g         value of the channel G
 * \param[in]  b         value of the channel B
 * \param[in]  index_best       index of the channel with highest color value
 * \param[in]  threshold 		if the difference of the color values is below this threshold, they are considered similar
 * 
 * max magnitude = 18080,358 (1024^(sqr(2))
 * **/
                         
/*unsigned char get_RGB_quantization(unsigned short r, unsigned short g, unsigned short b, unsigned short index_best, int threshold, bool& grayWhiteOrBlack)
{
	unsigned char response = -1;
	if(index_best == 0)
	{
		if(g+threshold >= r)
			if(b + threshold >= r)
			{ 
				if(grayEnabled == true)
				{
				    if(isBlack(r,g,b) == 0)//black
					    response = 128;
				    else if(isBlack(r,g,b) == 1)//white
					    response = 64;
				    else if(isBlack(r,g,b) == 2)//gray
				    {
					response = 0; //gray-black
					grayWhiteOrBlack = BLACK;
				    }
				    else //gray
				    {
					response = 0;//gray-white
					grayWhiteOrBlack= WHITE;
				    }
				}
				else //grayEnabled == false
				{
				    if(isBlack(r,g,b) == 0 || isBlack(r,g,b) == 2)//black
					    response = 128;
				    else //white
					response = 64;				
				}
			}
			else
				response = 8;
		else
			if(b + threshold >= r)
				response = 16;
			else
				response = 1;
	}
	else if(index_best == 1)
	{
		if(r+threshold >= g)
			if(b + threshold >= g)
			{
				if(grayEnabled == true)
				{
				    if(isBlack(r,g,b) == 0)//black
					    response = 128;
				    else if(isBlack(r,g,b) == 1)//white
					    response = 64;
				    else if(isBlack(r,g,b) == 2)//gray
				    {
					response = 0; //gray-black
					grayWhiteOrBlack = BLACK;
				    }
				    else //gray
				    {
					response = 0;//gray-white
					grayWhiteOrBlack= WHITE;
				    }
				}
				else
				{
				    if(isBlack(r,g,b) == 0 || isBlack(r,g,b) == 2)//black
					    response = 128;
				    else //white
					response = 64;				
				}
			}
			else
				response = 8;
		else
			if(b + threshold >= g)
				response = 32;
			else
				response = 2;
	}
	else if(index_best == 2)
	{
		//if(DEBUGGING) std::cout<<"rosso: "<<r<< " -  verde: "<<g<<" - blu: "<<b<<std::endl;
		if(r+threshold >= b)
			if(g + threshold >= b)
			{
				if(grayEnabled == true)
				{
				    if(isBlack(r,g,b) == 0)//black
					    response = 128;
				    else if(isBlack(r,g,b) == 1)//white
					    response = 64;
				    else if(isBlack(r,g,b) == 2)//gray
				    {
					response = 0; //gray-black
					grayWhiteOrBlack = BLACK;
				    }
				    else //gray
				    {
					response = 0;//gray-white
					grayWhiteOrBlack= WHITE;
				    }
				}
				else
				{
				    if(isBlack(r,g,b) == 0 || isBlack(r,g,b) == 2)//black
					    response = 128;
				    else //white
					response = 64;				
				}
			}
			else
				response = 16;
		else
			if(g + threshold >= b)
				response = 32;
			else
				response = 4;
	}
	//if(index_best == 1)
	//	if(DEBUGGING) std::cout<<"response rgb: "<<(int)response<<std::endl;
		
	return response;
}*/

unsigned char get_RGB_quantization(unsigned short r, unsigned short g, unsigned short b, unsigned short index_best, int threshold, bool& grayWhiteOrBlack)
{
	unsigned short lower_thresh = 50;
	unsigned short low_red_and_blu_thresh = 15;
	
	/*if(index_best == 0)
	    if(r<low_red_and_blu_thresh)
		return 128;
	if(index_best == 1)
	    if(g<low_red_and_blu_thresh)
		return 128;
	if(index_best == 2)
	    if(b<low_red_and_blu_thresh)
		return 128;
	*/
	if(index_best == 0 || index_best == 2)
	{
	    unsigned short tmp_c;
	    if(index_best  == 0)
		tmp_c = r;
	    if(index_best  == 2)
		tmp_c = b;
		
	    if(tmp_c<=lower_thresh)
		threshold = low_red_and_blu_thresh;
	}
    
	//abilita/disabilita
	//low_red_and_blu_thresh = threshold;
    
	unsigned char response = -1;
	if(index_best == 0)
	{
		if(g+threshold>= r)
			if(b + threshold >= r)
			{ 
				if(grayEnabled == true)
				{
				    if(isBlack(r,g,b) == 0)//black
					    response = 128;
				    else if(isBlack(r,g,b) == 1)//white
					    response = 64;
				    else if(isBlack(r,g,b) == 2)//gray
				    {
					response = 0; //gray-black
					grayWhiteOrBlack = BLACK;
				    }
				    else //gray
				    {
					response = 0;//gray-white
					grayWhiteOrBlack= WHITE;
				    }
				}
				else //grayEnabled == false
				{
				    if(isBlack(r,g,b) == 0 || isBlack(r,g,b) == 2)//black
					    response = 128;
				    else //white
					response = 64;				
				}
			}
			else
			
				response = 8;
		else
			if(b + threshold >= r)
				response = 16;
			else
				response = 1;
	}
	else if(index_best == 1)
	{
		if(r+threshold>= g)
			if(b + threshold >= g)
			{
				if(grayEnabled == true)
				{
				    if(isBlack(r,g,b) == 0)//black
					    response = 128;
				    else if(isBlack(r,g,b) == 1)//white
					    response = 64;
				    else if(isBlack(r,g,b) == 2)//gray
				    {
					response = 0; //gray-black
					grayWhiteOrBlack = BLACK;
				    }
				    else //gray
				    {
					response = 0;//gray-white
					grayWhiteOrBlack= WHITE;
				    }
				}
				else
				{
				    if(isBlack(r,g,b) == 0 || isBlack(r,g,b) == 2)//black
					    response = 128;
				    else //white
					response = 64;				
				}
			}
			else
				response = 8;
		else
			if(b + threshold >= g)
				response = 32;
			else
				response = 2;
	}
	else if(index_best == 2)
	{
		//if(DEBUGGING) std::cout<<"rosso: "<<r<< " -  verde: "<<g<<" - blu: "<<b<<std::endl;
		if(r+threshold >= b)
			if(g + threshold >= b)
			{
				if(grayEnabled == true)
				{
				    if(isBlack(r,g,b) == 0)//black
					    response = 128;
				    else if(isBlack(r,g,b) == 1)//white
					    response = 64;
				    else if(isBlack(r,g,b) == 2)//gray
				    {
					response = 0; //gray-black
					grayWhiteOrBlack = BLACK;
				    }
				    else //gray
				    {
					response = 0;//gray-white
					grayWhiteOrBlack= WHITE;
				    }
				}
				else
				{
				    if(isBlack(r,g,b) == 0 || isBlack(r,g,b) == 2)//black
					    response = 128;
				    else //white
					response = 64;				
				}
			}
			else
				response = 16;
		else
			if(g + threshold >= b)
				response = 32;
			else
				response = 4;
	}
	/*if(index_best == 1)
		if(DEBUGGING) std::cout<<"response rgb: "<<(int)response<<std::endl;
		*/
	return response;
}


/**
 * \brief Compute quantized orientation image from color image.
 *
 * Implements section 2.2 "Computing the Gradient Orientations."
 *
 * \param[in]  src       The source 8-bit, 3-channel image.
 * \param[out] magnitude Destination floating-point array of squared magnitudes.
 * \param[out] angle     Destination 8-bit array of orientations. Each bit
 *                       represents one bin of the orientation space.
 * \param      threshold Magnitude threshold. Keep only gradients whose norms are
 *                       larger than this.
 */
void quantizedOrientations(const Mat& src, Mat& magnitude, Mat& angle, Mat& rgb, Mat_<bool>& whiteOrBlack, float w_threshold, float s_threshold, bool computeMagnitudeStrong, Mat& magnitudeStrong)
{
//std::cout<<"weak: "<<w_threshold<<" - strong: "<<s_threshold<<endl;
    
  magnitude.create(src.size(), CV_32F);
  rgb.create(src.size(), CV_8UC1);
  //rawRgb.create(src.size(), CV_8UC1);
  whiteOrBlack.create(src.size());
  whiteOrBlack.setTo(WHITE);
  
  if(computeMagnitudeStrong == true)
  {
    magnitudeStrong.create(src.size(), CV_32F);
    magnitudeStrong.setTo(0);
  }  
  // Allocate temporary buffers
  Size size = src.size();

  Mat sobel_3dx; // per-channel horizontal derivative
  Mat sobel_3dy; // per-channel vertical derivative
  Mat sobel_dx(size, CV_32F);      // maximum horizontal derivative
  Mat sobel_dy(size, CV_32F);      // maximum vertical derivative
  Mat sobel_ag;  // final gradient orientation (unquantized)
  Mat smoothed;


  // Compute horizontal and vertical image derivatives on all color channels separately
  static const int KERNEL_SIZE = 3;
  
  // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
  GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, KERNEL_SIZE/*CV_SCHARR*/, 1.0, 0.0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, KERNEL_SIZE/*CV_SCHARR*/, 1.0, 0.0, BORDER_REPLICATE);

  /*Mat abs_grad_x, abs_grad_y, grad;
  /// Gradient X
  convertScaleAbs( sobel_3dy, abs_grad_x );
  /// Gradient Y
  convertScaleAbs( sobel_3dy, abs_grad_y );
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  imshow("sobel technicolor", grad );
  waitKey(0);
  */
  
  /*Mat mDebug0(sobel_3dx.rows,sobel_3dx.cols,CV_16S,Scalar(255));
  Mat mDebugAND0;
  
  mDebugAND0 = mDebug0 & sobel_3dx;
  cv::imshow("sobel subito dopo sobel 3dx: ",mDebugAND0);
  mDebugAND0 = mDebug0 & sobel_3dy;
  cv::imshow("sobel subito dopo sobel 3dy: ",mDebugAND0);
  */
  
  short * ptrx  = (short *)sobel_3dx.data;
  short * ptry  = (short *)sobel_3dy.data;
  float * ptr0x = (float *)sobel_dx.data;
  float * ptr0y = (float *)sobel_dy.data;
  float * ptrmg = (float *)magnitude.data;
  
  uchar * ptrrgb = (uchar *)rgb.data;
  //uchar * ptrrgb_mag = (uchar *)rawRgb.data;
  uchar * ptrsrc = (uchar *)src.data;

  const int length1 = static_cast<const int>(sobel_3dx.step1());
  const int length2 = static_cast<const int>(sobel_3dy.step1());
  const int length3 = static_cast<const int>(sobel_dx.step1());
  const int length4 = static_cast<const int>(sobel_dy.step1());
  const int length5 = static_cast<const int>(magnitude.step1());
  const int length0 = sobel_3dy.cols * 3;

  const int length7 = static_cast<const int>(rgb.step1());
  const int length8 = static_cast<const int>(src.step1());
  //const int length9 = static_cast<const int>(rawRgb.step1());
  
  int max = 0;
  int min = 0;
  
  for (int r = 0; r < sobel_3dy.rows; ++r)
  {
    int ind = 0;

    for (int i = 0; i < length0; i += 3)
    {
	
     /*if(DEBUGGING) std::cout<<"bluex: "<<ptrx[i]<<" - bluey: "<<ptry[i]<<std::endl;
     if(DEBUGGING) std::cout<<"greenx: "<<ptrx[i+1]<<" - greeny: "<<ptry[i+1]<<std::endl;
     if(DEBUGGING) std::cout<<"redx: "<<ptrx[i+2]<<" - redy: "<<ptry[i+2]<<std::endl;*/
     /*if(ptrx[i] > max)
	max = ptrx[i];
    if(ptrx[i] < min)
	min = ptrx[i];
     */
     
      // Use the gradient orientation of the channel whose magnitude is largest
	  int mag1 = CV_SQR(ptrx[i]) + CV_SQR(ptry[i]);
      int mag2 = CV_SQR(ptrx[i + 1]) + CV_SQR(ptry[i + 1]);
      int mag3 = CV_SQR(ptrx[i + 2]) + CV_SQR(ptry[i + 2]);

//if(DEBUGGING) std::cout<<"mag1: "<<mag1<<" - mag2: "<<mag2<<" - mag3: "<<mag3<<std::endl;

      if (mag1 >= mag2 && mag1 >= mag3)
      {
		
        ptr0x[ind] = ptrx[i];
        ptr0y[ind] = ptry[i];
        ptrmg[ind] = (float)mag1;
        
	//float thresh = ((float)(mag1/100))*threshold_mag;
	//ptrrgb_mag[ind] = get_magnitude_RGB_quantization((float)mag3, (float)mag2, (float)mag1, 2, thresh);
        
	//mag3 and mag1 are inverted because mats are bgr instead rgb
        //ptrrgb[ind] = get_RGB_quantization(ptrsrc[i+2], ptrsrc[i+1], ptrsrc[i], 0, threshold_rgb);
        //ptrmgrgb[ind] = get_magnitude_RGB_quantization(mag3, mag2, mag1, ptrsrc[i+2], ptrsrc[i+1], ptrsrc[i], 0, threshold_rgb);
      }
      else if (mag2 >= mag1 && mag2 >= mag3)
      {
        ptr0x[ind] = ptrx[i + 1];
        ptr0y[ind] = ptry[i + 1];
        ptrmg[ind] = (float)mag2;
        
	//float thresh = ((float)(mag2/100))*threshold_mag;
	//ptrrgb_mag[ind] = get_magnitude_RGB_quantization((float)mag3, (float)mag2, (float)mag1, 1, thresh);
        
	//ptrrgb[ind] = get_RGB_quantization(ptrsrc[i+2], ptrsrc[i+1], ptrsrc[i], 1, threshold_rgb);
        //ptrmgrgb[ind] = get_magnitude_RGB_quantization(mag3, mag2, mag1, ptrsrc[i+2], ptrsrc[i+1], ptrsrc[i], 1, threshold_rgb);
      }
      else
      {
        ptr0x[ind] = ptrx[i + 2];
        ptr0y[ind] = ptry[i + 2];
        ptrmg[ind] = (float)mag3;
	
	//float thresh = ((float)(mag3/100))*threshold_mag;
	//ptrrgb_mag[ind] = get_magnitude_RGB_quantization((float)mag3, (float)mag2, (float)mag1, 0, thresh);
        
        //ptrrgb[ind] = get_RGB_quantization(ptrsrc[i+2], ptrsrc[i+1], ptrsrc[i], 2, threshold_rgb);
        //ptrmgrgb[ind] = get_magnitude_RGB_quantization(mag3, mag2, mag1, ptrsrc[i+2], ptrsrc[i+1], ptrsrc[i], 2, threshold_rgb);
      }
      

		  unsigned short red = (unsigned short)ptrsrc[i+2];
		  unsigned short green = (unsigned short)ptrsrc[i+1];
		  unsigned short blue = (unsigned short)ptrsrc[i];
		  
		  if(red >= green && red >= blue)
				ptrrgb[ind] = get_RGB_quantization(red, green, blue, 0, threshold_rgb, whiteOrBlack[r][ind]);
		  else if(green >= red && green >= blue)
				ptrrgb[ind] = get_RGB_quantization(red, green, blue, 1, threshold_rgb, whiteOrBlack[r][ind]);
		  else
				ptrrgb[ind] = get_RGB_quantization(red, green, blue, 2, threshold_rgb, whiteOrBlack[r][ind]);
				

		  
		  
		  /*
		  if(ptrmg[ind] >= CV_SQR(s_threshold))
		  {
		      if(DEBUGGING) std::cout<<std::endl<<"magTotal: "<<ptrmg[ind]<<std::endl;
		      if(DEBUGGING) std::cout<<"magRed: "<<mag3<<std::endl;
		      if(DEBUGGING) std::cout<<"magGreen: "<<mag2<<std::endl;
		      if(DEBUGGING) std::cout<<"magBlue: "<<mag1<<std::endl;
		      if(DEBUGGING) std::cout<<"risultato: "<<getColorFromLabel(ptrrgb_mag[ind])<<std::endl<<std::endl;
		  }
		   */
      
      ++ind;
    }
    ptrx += length1;
    ptry += length2;
    ptr0x += length3;
    ptr0y += length4;
    ptrmg += length5;
    
    //ptrmgrgb += length6;
    ptrrgb += length7;
    ptrsrc += length8;
    //ptrrgb_mag += length9;
  }
  
  //if(DEBUGGING) std::cout<<"MAX: "<<max<<std::endl;
  //if(DEBUGGING) std::cout<<"MIN: "<<min<<std::endl;

/*
if(onlyOne)
{
  Mat mDebug(src.rows,src.cols,CV_32F,Scalar(255));
  Mat mDebugAND;
  cv::imshow("src: ",src);
  if(DEBUGGING) std::cout<<"src.rows: "<<src.rows<<" - src.cols: "<<src.cols<<std::endl;
  if(DEBUGGING) std::cout<<"src type: "<<src.type()<<" type 32f: "<<CV_32F<<" type 32s: "<<CV_32S<<" type 64F: "<<CV_64F<<std::endl;
  
  for(int r = 0; r<src.rows; r++)
	for(int c = 0; c<src.cols; c++)
	{
		if((int)src.at<Vec3b>(r,c)[2] == 0)
		{
		if(DEBUGGING) std::cout<<"B - mat["<<r<<"]["<<c<<"]: "<<(int)src.at<Vec3b>(r,c)[0]<<" ";
		if(DEBUGGING) std::cout<<"G - mat["<<r<<"]["<<c<<"]: "<<(int)src.at<Vec3b>(r,c)[1]<<" ";
		if(DEBUGGING) std::cout<<"R - mat["<<r<<"]["<<c<<"]: "<<(int)src.at<Vec3b>(r,c)[2]<<std::endl;
		}
	}
	
  mDebugAND = mDebug & sobel_dx;
  cv::imshow("sobel dx: ",mDebugAND);
  mDebugAND = mDebug & sobel_dy;
  cv::imshow("sobel dy: ",mDebugAND);
  onlyOne = false;
}*/
//if(DEBUGGING) std::cout<<"quant5"<<std::endl;  
  // Calculate the final gradient orientations
  phase(sobel_dx, sobel_dy, sobel_ag, true);
  
  
/*if(onlyOne_1)
{
  Mat mDebug(src.rows,src.cols,CV_32F,Scalar(255));
  Mat mDebugAND;
  mDebugAND = mDebug & sobel_ag;
  cv::imshow("debug prima dell'isteresi",mDebugAND);
  onlyOne_1 = false;
}*/
  
  
  //cv::imshow("debug magnitude prima isteresi",(1/magnitude)*255);
  
  hysteresisGradient(magnitude, angle, sobel_ag, CV_SQR(w_threshold), CV_SQR(s_threshold), computeMagnitudeStrong, magnitudeStrong);
  Mat mag_temp;
  if(DEBUGGING)
  {
      convertScaleAbs( magnitude/100, mag_temp);
      imshow("magnitude in quantized", mag_temp);
      imshow("angle in quantized", angle);
    }
  
  
  if(DEBUGGING)
  {
      Mat srcTemp;
      src.copyTo(srcTemp);
      for (int lr = 0; lr < rgb.rows; ++lr)
	{
	    const uchar* rgb_tmp_r = rgb.ptr<uchar>(lr); 
	    for (int lc = 0; lc < rgb.cols; ++lc)
	    {
	      cv::Scalar colorT;
	      cv::Point pt(lc, lr);
	      switch(getLabel(rgb_tmp_r[lc]))
	      {
	      case 0: colorT = CV_RGB(255,0,0); break;
	      case 1: colorT = CV_RGB(0,255,0); break;
	      case 2: colorT = CV_RGB(0,0,255); break;
	      case 3: colorT = CV_RGB(255,255,0); break;
	      case 4: colorT = CV_RGB(255,0,255); break;
	      case 5: colorT = CV_RGB(0,255,255); break;
	      case 6: colorT = CV_RGB(255,255,255); break;
	      case 7: colorT = CV_RGB(0,0,0); break;
	      case 8: colorT = CV_RGB(127,127,127); break;
	      case 9: colorT = CV_RGB(127,127,127); break;
	      }
	      cv::circle(srcTemp, pt, 0, colorT);
	    }
	}
	if(computeMagnitudeStrong == true)
	    imshow("colorRGB_quant", srcTemp);	
   } 
    
  
  
 /* imshow("sobel_x in quantized", (1/sobel_3dx)*255);
  imshow("sobel_y in quantized", (1/sobel_3dy)*255);
  imshow("magnitude in quantized", (1/magnitude)*255);
  imshow("sobel_ag in quantized", (1/sobel_ag)*255);
  
  waitKey();
  */
/*if(onlyOne_2)
{
  if(DEBUGGING) std::cout<<"src.rows: "<<src.rows<<" - src.cols: "<<src.cols<<std::endl;
  if(DEBUGGING) std::cout<<"sobel_ag.rows: "<<sobel_ag.rows<<" - sobel_ag.cols: "<<sobel_ag.cols<<std::endl;
  if(DEBUGGING) std::cout<<"angle.rows: "<<angle.rows<<" - angle.cols: "<<angle.cols<<std::endl;
  if(DEBUGGING) std::cout<<"magnitude.rows: "<<magnitude.rows<<" - magnitude.cols: "<<magnitude.cols<<std::endl;
	//if(DEBUGGING) std::cout<<"magnitude matrice: "<<magnitude<<std::endl;
	//if(DEBUGGING) std::cout<<"angoli matrice: "<<angle<<std::endl;
	
	
  Mat mDebug(src.rows,src.cols,CV_32F,Scalar(255));
  Mat mDebugAND;
  mDebugAND = mDebug & sobel_ag;
  cv::imshow("debug dopo dell'isteresi",angle);
  onlyOne_2 = false;
}*/


//cv::imshow("debug magnitude dopo isteresi",(1/magnitude)*255);
//waitKey(0);
  /*cv::imshow("debug",src);
  
  mDebugAND = mDebug & sobel_ag;
  cv::imshow("debug dopo l'isteresi",(1/mDebugAND)*255);
  cv::imshow("debug magnitude",(1/magnitude)*255);
  
  //if(DEBUGGING) std::cout<<"matrice: "<<sobel_ag<<std::endl;
  if(DEBUGGING) std::cout<<"src.cols: "<<src.cols<<" src.rows: "<<src.rows<<" - sobel_ag.cols: "<<sobel_ag.cols<<" sobel_ag.rows: "<<sobel_ag.rows<<std::endl;
  if(DEBUGGING) std::cout<<"mDebug.cols: "<<mDebug.cols<<" mDebug.rows: "<<mDebug.rows<<std::endl;
  cv::waitKey(0);
  */
  
	  /*Mat mDebug0(sobel_3dx.rows,sobel_3dx.cols,sobel_3dx.type(),Scalar(255));
	  Mat mDebugAND0;
	  mDebugAND0 = mDebug0 & sobel_3dx;
	  //cv::imshow("sobel subito dopo sobel 3dx: ",mDebugAND0);
	  cv::imshow("sobel subito dopo sobel 3dx: ",sobel_3dx);
	  mDebugAND0 = mDebug0 & sobel_3dy;
	  //cv::imshow("sobel subito dopo sobel 3dy: ",mDebugAND0);
	  cv::imshow("sobel subito dopo sobel 3dy: ",sobel_3dy);
	  */
	  
	  /*
	  imshow("sobel_x in quantized", (1/sobel_3dx)*2550);
	  imshow("sobel_y in quantized", (1/sobel_3dy)*2550);
	  
	  Mat mDebug(src.rows,src.cols,CV_32F,Scalar(255));
          Mat mDebugAND;
	  mDebugAND = mDebug & sobel_dx;
	  cv::imshow("sobel dx: ",mDebugAND);
	  mDebugAND = mDebug & sobel_dy;
	  cv::imshow("sobel dy: ",mDebugAND);
	  cv::imshow("src",(src));
	  cv::imshow("RGB",(rgb));
	  cv::imshow("angle",angle);
	  waitKey(0);
	  */
}

void hysteresisGradient(Mat& magnitude, Mat& quantized_angle,
                        Mat& angle, float w_threshold, float s_threshold, bool computeMagnitudeStrong, Mat& magnitudeStrong)
{
    //Mat tmpMag;
    //cv::threshold(magnitude, tmpMag, 3000.0f, 255, THRESH_TOZERO);
    //imshow("tempMAg", tmpMag);
    //waitKey();
    
  // Quantize 360 degree range of orientations into 16 buckets
  // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
  // for stability of horizontal and vertical features.
  Mat_<unsigned char> quantized_unfiltered;
  //if(DEBUGGING) std::cout<<"in histeresis angle.rows: "<<angle.rows<<" - angle.cols: "<<angle.cols<<std::endl;
  angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

  // Zero out top and bottom rows
  /// @todo is this necessary, or even correct?
  memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
  memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
  // Zero out first and last columns
  for (int r = 0; r < quantized_unfiltered.rows; ++r)
  {
    quantized_unfiltered(r, 0) = 0;
    quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
  }

  // Mask 16 buckets into 8 quantized orientations
  for (int r = 1; r < angle.rows - 1; ++r)
  {
    uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
    for (int c = 1; c < angle.cols - 1; ++c)
    {
      quant_r[c] &= 7;
    }
  }

  // Filter the raw quantized image. Only accept pixels where the magnitude is above some
  // threshold, and there is local agreement on the quantization.
 
 int magCount = 0;
 int magStrongCount = 0;
 
  quantized_angle = Mat::zeros(angle.size(), CV_8U);
  for (int r = 1; r < angle.rows - 1; ++r)
  {
    float* mag_r = magnitude.ptr<float>(r);

    for (int c = 1; c < angle.cols - 1; ++c)
    {
	
    
      if (mag_r[c] > w_threshold)
      {
	  magCount++;
	// Compute histogram of quantized bins in 3x3 patch around pixel
        int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        uchar* patch3x3_row = &quantized_unfiltered(r-1, c-1);
        histogram[patch3x3_row[0]]++;
        histogram[patch3x3_row[1]]++;
        histogram[patch3x3_row[2]]++;

	patch3x3_row += quantized_unfiltered.step1();
        histogram[patch3x3_row[0]]++;
        histogram[patch3x3_row[1]]++;
        histogram[patch3x3_row[2]]++;

	patch3x3_row += quantized_unfiltered.step1();
        histogram[patch3x3_row[0]]++;
        histogram[patch3x3_row[1]]++;
        histogram[patch3x3_row[2]]++;

	// Find bin with the most votes from the patch
        int max_votes = 0;
        int index = -1;
        for (int i = 0; i < 8; ++i)
        {
          if (max_votes < histogram[i])
          {
            index = i;
            max_votes = histogram[i];
          }
        }

	if(computeMagnitudeStrong == true && mag_r[c]> s_threshold)
	{
	    magnitudeStrong.at<float>(r, c) = mag_r[c];
	    magStrongCount++;
	}

	// Only accept the quantization if majority of pixels in the patch agree
	static const int NEIGHBOR_THRESHOLD = 5;
        if (max_votes >= NEIGHBOR_THRESHOLD)
	{
          quantized_angle.at<uchar>(r, c) = 1 << index;
	  
        }
      }
    }
  }
  if(DEBUGGING) std::cout<<"computeMagnitudeStrong: "<<computeMagnitudeStrong<<std::endl;
  if(DEBUGGING) std::cout<<"magCount: "<<magCount<<std::endl;
if(DEBUGGING) std::cout<<"magStrongCount: "<<magStrongCount<<std::endl;

}

class ColorGradientPyramid : public QuantizedPyramid
{
public:
  ColorGradientPyramid(const Mat& src, const Mat& mask,
                       float weak_threshold, size_t num_features,
                       float strong_threshold);

  virtual void quantize(Mat& dst) const;
  
  virtual void quantizeRGB(Mat& dst) const;

  virtual bool extractTemplate(Template& templ) const;

  virtual void pyrDown();

  Mat magnitudeStrong;
  Mat magnitude;
protected:
  /// Recalculate angle and magnitude images
  void update(bool computeMagnitudeStrong);

  Mat src;
  Mat mask;

  int pyramid_level;
  Mat angle;
  
  Mat_<bool> whiteOrBlack;
  Mat rgb;
  

  float weak_threshold;
  size_t num_features;
  float strong_threshold;
};

ColorGradientPyramid::ColorGradientPyramid(const Mat& src, const Mat& mask,
                                           float weak_threshold, size_t num_features,
                                           float strong_threshold)
  : src(src),
    mask(mask),
    pyramid_level(0),
    weak_threshold(weak_threshold),
    num_features(num_features),
    strong_threshold(strong_threshold)
{
  update(true);
}

void ColorGradientPyramid::update(bool computeMagnitudeStrong)
{ 
  quantizedOrientations(src, magnitude, angle, rgb, whiteOrBlack, weak_threshold, strong_threshold, computeMagnitudeStrong, magnitudeStrong);
}

void ColorGradientPyramid::pyrDown()
{
	
  // Some parameters need to be adjusted
  num_features /= 2; /// @todo Why not 4?
  ++pyramid_level;

  // Downsample the current inputs
  Size size(src.cols / 2, src.rows / 2);
  Mat next_src;
  cv::pyrDown(src, next_src, size);
  src = next_src;
  if (!mask.empty())
  {
    Mat next_mask;
    resize(mask, next_mask, size, 0.0, 0.0, CV_INTER_NN);
    mask = next_mask;
  }

  update(false);

}

void ColorGradientPyramid::quantize(Mat& dst) const
{
  dst = Mat::zeros(angle.size(), CV_8U);
  angle.copyTo(dst, mask);
}

void ColorGradientPyramid::quantizeRGB(Mat& dst) const
{

      dst = Mat::zeros(rgb.size(), CV_8UC1);
      rgb.copyTo(dst, mask);
  
}

//conto i valori rgb dei vicini e controllo anche quanti di essi sono nella maschera
int fillRgbNeighbours(int cBegin, int cEnd, int rBegin, int rEnd, const Mat& rgb, const Mat_<bool>& whiteOrBlack, const Mat& magnitude, const Mat& mask, int* rgbValues, float magMin)
{
    int countInMask = 0;
    
    for(int i = rBegin; i<=rEnd; i++)
    {
	const float* magnitude_r = magnitude.ptr<float>(i);
	const uchar* rgb_r = rgb.ptr<uchar>(i);
	const uchar* mask_r = mask.ptr<uchar>(i);
	
	for(int j = cBegin; j<=cEnd; j++)
	{		
	    //if(DEBUGGING) std::cout<<"["<<(int)rgb_r[j]<<"]";
	    if(mask_r[j])
		countInMask++;
	    	    
	    if(magnitude_r[j] >= magMin)
	    {
		if(rgb_r[j] == 0)
		{
		    if(whiteOrBlack[i][j] == WHITE)
			rgbValues[6]++;
		    else if(whiteOrBlack[i][j] == BLACK)
			rgbValues[7]++;
		}
		else
		    rgbValues[(int)log2((double)rgb_r[j])]++;
	    }
	}
	
	//if(DEBUGGING) std::cout<<std::endl;
	    	
    }
    
    return countInMask;
}


void calcCR(int c, int r, const Mat& magnitude, uchar pAngle, int offsetPlusMinus, int& cBegin, int& cEnd, int& rBegin, int& rEnd, int kS)
{
    //primi 8, valori di x o y positivi (maggior di centralMass), ulimi 8, valori negativi
    int offsetRInit[16] = {/*1+*/-(kS/2),/*2+*/-(kS/2),/*4+*/0,/*8+*/0,/*16+*/0,/*32+*/0,/*64+*/0,/*128+*/-(kS/2),/*1-*/-(kS/2),/*2-*/-(kS/2),/*4-*/-kS+1,/*8-*/-kS+1,/*16-*/-kS+1,/*32-*/-kS+1,/*64-*/-kS+1,/*128-*/-(kS/2)};
    int offsetCInit[16] = {/*1+*/-kS+1,/*2+*/-kS+1,/*4+*/0,/*8+*/-(kS/2),/*16+*/-(kS/2),/*32+*/-(kS/2),/*64+*/-kS+1,/*128+*/-kS+1,/*1-*/0,/*2-*/0,/*4-*/-kS+1,/*8-*/-(kS/2),/*16-*/-(kS/2),/*32-*/-(kS/2),/*64-*/0,/*128-*/0};
    int offsetREnd[16] = {/*1+*/kS/2,/*2+*/kS/2,/*4+*/kS-1,/*8+*/kS-1,/*16+*/kS-1,/*32+*/kS-1,/*64+*/kS-1,/*128+*/kS/2,/*1-*/kS/2,/*2-*/kS/2,/*4-*/0,/*8-*/0,/*16-*/0,/*32-*/0,/*64-*/0,/*128-*/kS/2};
    int offsetCEnd[16] = {/*1+*/0,/*2+*/0,/*4+*/kS-1,/*8+*/kS/2,/*16+*/kS/2,/*32+*/kS/2,/*64+*/0,/*128+*/0,/*1-*/kS-1,/*2-*/kS-1,/*4-*/0,/*8-*/kS/2,/*16-*/kS/2,/*32-*/kS/2,/*64-*/kS-1,/*128-*/kS-1};
    
    int quantAngle = getLabel(pAngle);
    
    cBegin = c + offsetCInit[quantAngle+offsetPlusMinus];
    if(cBegin < 0)
	cBegin = 0;
    cEnd = c + offsetCEnd[quantAngle+offsetPlusMinus];
    if(cEnd > magnitude.cols)
	cEnd = magnitude.cols;
    rBegin = r +offsetRInit[quantAngle+offsetPlusMinus];
    if(rBegin < 0)
	rBegin = 0;
    rEnd = r + offsetREnd[quantAngle+offsetPlusMinus];
    if(rEnd > magnitude.rows)
	rEnd = magnitude.rows;
	
    /*int testSquare [25][25] = {0};
    
    int cs[16] = {24,24,0,12,12,12,24,24, 0,0,24,12,12,12,0,0};
    int rs[16] = {12,12,0,0,0,0,0,12, 12,12,24,24,24,24,24,12};
    
    for(int j =0; j< 8; j++)
    {
	int cbegin_plus = cs[j] + offsetCInit[j];
	int cend_plus = cs[j]+offsetCEnd[j];
	int rbegin_plus = rs[j]+offsetRInit[j];
	int rend_plus = rs[j]+offsetREnd[j];
	
	int cbegin_minus = cs[j+8]+offsetCInit[j+8];
	int cend_minus = cs[j+8]+offsetCEnd[j+8];
	int rbegin_minus = rs[j+8]+offsetRInit[j+8];
	int rend_minus = rs[j+8]+offsetREnd[j+8];
	
	for(int hr = rbegin_plus; hr<=rend_plus; hr++)
	    for(int hc = cbegin_plus; hc<=cend_plus; hc++)
		testSquare[hr][hc]++;
	for(int hr = rbegin_minus; hr<=rend_minus; hr++)
	    for(int hc = cbegin_minus; hc<=cend_minus; hc++)
		testSquare[hr][hc]++;
    }
    
    for(int hr = 0; hr<25; hr++)
    {
	for(int hc = 0; hc<25; hc++)
	    if(DEBUGGING) std::cout<<"["<<testSquare[hr][hc]<<"]";
	if(DEBUGGING) std::cout<<std::endl;
    }
    imshow("rgb per fermare", rgb);
    waitKey(0);
    */
}


uchar votesRgbMag(int c, int r, float magCenter, uchar pAngle, const Mat& rgb, const Mat_<bool>& whiteOrBlack, const Mat& magnitude, const Mat& mask, Point centralMass)
{
    float magThresh = 100;
    float magMin = magCenter - (magCenter/100)*magThresh;
    
    int rgbValues[8] = {0,0,0,0,0,0,0,0};
    
    //kernelSize (apertura vicini)
    int kernelSize = 5;
    
    //calcolo se devo provare con l'angolo positivo o negativo
    int offsetPlusMinus = 0;
    if(pAngle == 1 || pAngle == 2 || pAngle == 128)
    {
	if(c >= (int)centralMass.x)
	    offsetPlusMinus = 0;
	else
	    offsetPlusMinus = 8;
    }
    else if(pAngle == 8 || pAngle == 16 || pAngle == 32 || pAngle == 64 || pAngle == 4)
    {
	if(r <= (int)centralMass.y)
	    offsetPlusMinus = 0;
	else
	    offsetPlusMinus = 8;
    }
    else
    {
	CV_Assert(false);
    }
    
    int cBegin, cEnd, rBegin, rEnd;
    calcCR(c, r, magnitude, pAngle, offsetPlusMinus, cBegin, cEnd, rBegin, rEnd, kernelSize);
    
    int countInMask = fillRgbNeighbours(cBegin, cEnd, rBegin, rEnd, rgb, whiteOrBlack, magnitude, mask, rgbValues, magMin);
    
    //controllo se abbastanza vicini erano nella maschera
    int neighThresh = ((kernelSize * kernelSize)/2)+1;
    if(countInMask < neighThresh)
    {
		
	if(offsetPlusMinus == 0)
	    offsetPlusMinus = 8;
	else
	    offsetPlusMinus = 0;
	
	calcCR(c, r, magnitude, pAngle, offsetPlusMinus, cBegin, cEnd, rBegin, rEnd, kernelSize);
	
	for(int g = 0; g<8; g++)
	    rgbValues[g] = 0;
	
	countInMask = fillRgbNeighbours(cBegin, cEnd, rBegin, rEnd, rgb, whiteOrBlack, magnitude, mask, rgbValues, magMin);
	
	//se non abbiamo ancora abbastanza vicini nella maschera, ritorno il valore di partenza
	if(countInMask < neighThresh)
	{
	    const uchar* rgb_r = rgb.ptr<uchar>(r);
	    return rgb_r[c];
	}
    }
    
    /*if(DEBUGGING) std::cout<<"rgbValues: ";
    for(int i = 0; i<8; i++)
	if(DEBUGGING) std::cout<<"["<<rgbValues[i]<<"]";
    */
    
    int max_votes = 0;
    int index = -1;
    for(int i = 0; i<8; i++)
    {
	if(rgbValues[i] > max_votes)
	{
	    max_votes = rgbValues[i];
	    index = i;
	}
    }
    
    uchar result;
    if(index != -1)
	result = (uchar) pow(2, index);
    else
    {
	const uchar* rgb_r = rgb.ptr<uchar>(r);
	result = rgb_r[c];
    }
	
    //if(DEBUGGING) std::cout<<std::endl<<"valore scelto: "<<(int)result<<std::endl;
    
    return result;

}

bool ColorGradientPyramid::extractTemplate(Template& templ) const
{
	
	
  bool onlyBorder = false;
  
  // Want features on the border to distinguish from background
  Mat local_mask;
  
  Mat border_mask;
  if (!mask.empty())
  {
	if(onlyBorder == true)
	{
		erode(mask, local_mask, Mat(), Point(-1,-1), 1, BORDER_REPLICATE);
		subtract(mask, local_mask, local_mask);
	}
	else
		mask.copyTo(local_mask);
	    //dilate(mask, local_mask, Mat(), Point(-1,-1), 3, BORDER_REPLICATE);
	erode(mask, border_mask, Mat(), Point(-1,-1), 3, BORDER_REPLICATE);
	subtract(mask, border_mask, border_mask);
    
  }
//  imshow("local_mask", local_mask);
//  imshow("border_mask", border_mask);


  // Create sorted list of all pixels with magnitude greater than a threshold
  std::vector<Candidate> candidates;
  bool no_mask = local_mask.empty();
  float threshold_sq = CV_SQR(strong_threshold);
  
  //imshow("magnitudine in EXTRACT TEMPLATE", (1/magnitude)*255);
  int quantCount = 0;
  int quantCountThreshold = 0;
    
  Moments moms = moments(mask, true);
  Point centralMass(moms.m10/moms.m00,moms.m01/moms.m00);
  float maskArea = moms.m00;
 /*
  Mat maskTmp;
  mask.copyTo(maskTmp);
  circle(maskTmp, centralMass, 4, Scalar(0,0,0));
  imshow("mascheramomentata", maskTmp);
  waitKey(0);
  */


  
  int stepSign = (int)sqrt(maskArea/(float)signFeat);
  
  for (int r = 0; r < magnitude.rows; ++r)
  {
    const uchar* angle_r = angle.ptr<uchar>(r);
    const uchar* rgb_r = rgb.ptr<uchar>(r);
    //const uchar* mag_rgb_r = rawRgb.ptr<uchar>(r);
    
    const float* magnitude_r = magnitude.ptr<float>(r);
    const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);
    const uchar* border_mask_r = no_mask ? NULL : border_mask.ptr<uchar>(r);

	//if(DEBUGGING) std::cout<<"IN EXTRACT TEMPLATE magnitude.rows: "<<magnitude.rows<<" - magnitude.cols: "<<magnitude.cols<<std::endl;

    

    
    for (int c = 0; c < magnitude.cols; ++c)
    {

      if (no_mask || mask_r[c] )//|| true == true)
      {
        uchar quantized = angle_r[c];
        uchar quantizedRgb = rgb_r[c];
	//uchar quantizedMagRgb = mag_rgb_r[c];
	
	if (magnitude_r[c] > threshold_sq)
	{
	    quantCount++;
	}
	
	if(r%stepSign == 0 && c%stepSign == 0 && !border_mask_r[c])
	{
	    templ.featuresSignature.push_back(cv::my_linemod::Feature(c, r, -1, getLabel(isGrayWhite(quantizedRgb, whiteOrBlack[r][c])), false));
	}

        if (quantized > 0)
        {
	  
          float score = magnitude_r[c];
	  //if(DEBUGGING) std::cout<<"threshold: "<<threshold_sq<<"- maag: "<<magnitude_r[c]<<std::endl;
          if (score > threshold_sq)
          {
	    bool sobelBorderFeatures = true;
	    

	    if(sobelBorderFeatures == true)
	    {
		quantCountThreshold++;
			  //if(DEBUGGING) std::cout<<"score: "<<score<<" - threshold_sq: "<<threshold_sq<<std::endl;
			  //if(DEBUGGING) std::cout<<"quantized magnitude: "<<(int)quantizedMagnitude<<std::endl;
	    
	    //
	    
	    
		if(border_mask_r[c])
		{
		    //votazione dei colori vicini
		    uchar voted = votesRgbMag(c,r,score, quantized, rgb, whiteOrBlack, magnitude, mask, centralMass);
		    candidates.push_back(Candidate(c, r, getLabel(quantized), getLabel(isGrayWhite(voted, whiteOrBlack[r][c])), true, score));
		}
		else
		    candidates.push_back(Candidate(c, r, getLabel(quantized), getLabel(isGrayWhite(quantizedRgb, whiteOrBlack[r][c])), false, score));
	    }
	    else
	    {
		quantCountThreshold++;
		
		if(border_mask_r[c])
		    candidates.push_back(Candidate(c, r, getLabel(quantized), getLabel(isGrayWhite(quantizedRgb, whiteOrBlack[r][c])), true, score));
		else
		    candidates.push_back(Candidate(c, r, getLabel(quantized), getLabel(isGrayWhite(quantizedRgb, whiteOrBlack[r][c])), false, score));
	    }
	    
          }
        }
      }
    }
    
  }
  
    Mat local_mask_temp(local_mask.cols, local_mask.rows, CV_8UC3, CV_RGB(100,100,100));
    //local_mask.copyTo(local_mask_temp);
    //cvtColor(local_mask_temp, local_mask_temp, CV_GRAY2RGB);
    
    if(DEBUGGING)
    {
	for (int itS = 0; itS < templ.featuresSignature.size(); itS++)
	{
	    cv::my_linemod::Feature featS = templ.featuresSignature.at(itS);
	    
	    cv::Scalar colorT;
	    cv::Point pt(featS.x, featS.y);
	    switch(featS.rgbLabel)
	    {
		case 0: colorT = CV_RGB(255,0,0); break;
		case 1: colorT = CV_RGB(0,255,0); break;
		case 2: colorT = CV_RGB(0,0,255); break;
		case 3: colorT = CV_RGB(255,255,0); break;
		case 4: colorT = CV_RGB(255,0,255); break;
		case 5: colorT = CV_RGB(0,255,255); break;
		case 6: colorT = CV_RGB(255,255,255); break;
		case 7: colorT = CV_RGB(0,0,0); break;
	    }
	    cv::circle(local_mask_temp, pt, 0, colorT);
	
	}
	imshow("mask senza features signature"+increment,local_mask);
	imshow("mask con features signature"+increment,local_mask_temp);
	increment = "small";
	std::cout<<"num features signature: "<<templ.featuresSignature.size()<<std::endl;
  }
  
  if(DEBUGGING) std::cout<<"quantCount: "<<quantCount<<std::endl;
  if(DEBUGGING) std::cout<<"quantCountThreshold: "<<quantCountThreshold<<std::endl;
  //templ.totalFeatures = candidates.size();
  templ.totalFeatures = quantCount;
  if(DEBUGGING) std::cout<<"templ.totalFeatures: "<<templ.totalFeatures<<std::endl;
  
  if(DEBUGGING) std::cout<<"num features: "<<num_features<<std::endl;
  
  // We require a certain number of features
  if (candidates.size() < num_features)
    return false;
  // NOTE: Stable sort to agree with old code, which used std::list::sort()
  std::stable_sort(candidates.begin(), candidates.end());


//if(DEBUGGING) std::cout<<"candidates.size: "<<candidates.size()<<std::endl;
//if(DEBUGGING) std::cout<<"magnitude.size: "<<magnitude.cols*magnitude.rows<<std::endl;
  if(DEBUGGING)
  {
      Mat tmp(angle.size(), CV_8UC1, Scalar(255));
      for(int k = 0; k<(int)candidates.size(); k++)
      {
	      
	      Candidate c = candidates.at(k);
	      Feature f = c.f;
	      tmp.at<uchar>(f.y,f.x) = f.rgbLabel;
	      
      }
      imshow("in EXTRACTTEMPLATE:",tmp);
  }
  
  
  // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
  float distance = static_cast<float>(candidates.size() / num_features + 1);
  selectScatteredFeatures(candidates, templ.features, num_features, distance);

  if(DEBUGGING)
  {
      Mat tmp2(angle.size(), CV_8UC1, Scalar(255));
      for(int k = 0; k<(int)templ.features.size(); k++)
      {
	      Feature f = templ.features[k];
	      tmp2.at<uchar>(f.y,f.x) = f.rgbLabel;
	      //if(DEBUGGING) std::cout<<"y: "<<f.y<<std::endl;
      }
      imshow("in EXTRACTTEMPLATE dopo lo scattered:",tmp2);
  }

  // Size determined externally, needs to match templates for other modalities
  templ.width = -1;
  templ.height = -1;
  templ.pyramid_level = pyramid_level;
//waitKey(0);
  return true;
}

ColorGradient::ColorGradient()
  : weak_threshold(50.0f),
    num_features(featuresUsed),
    strong_threshold(70.0f)
{
}

ColorGradient::ColorGradient(float weak_threshold, size_t num_features, float strong_threshold)
  : weak_threshold(weak_threshold),
    num_features(num_features),
    strong_threshold(strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
  return CG_NAME;
}

Ptr<QuantizedPyramid> ColorGradient::processImpl(const Mat& src,
                                                     const Mat& mask) const
{
  return new ColorGradientPyramid(src, mask, weak_threshold, num_features, strong_threshold);
}

void ColorGradient::read(const FileNode& fn)
{
  std::string type = fn["type"];
  CV_Assert(type == CG_NAME);

  weak_threshold = fn["weak_threshold"];
  num_features = int(fn["num_features"]);
  strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage& fs) const
{
  fs << "type" << CG_NAME;
  fs << "weak_threshold" << weak_threshold;
  fs << "num_features" << int(num_features);
  fs << "strong_threshold" << strong_threshold;
}

/****************************************************************************************\
*                               Depth normal modality                                    *
\****************************************************************************************/

// Contains GRANULARITY and NORMAL_LUT
#include "normal_lut.i"

static void accumBilateral(long delta, long i, long j, long * A, long * b, int threshold)
{
  long f = std::abs(delta) < threshold ? 1 : 0;

  const long fi = f * i;
  const long fj = f * j;

  A[0] += fi * i;
  A[1] += fi * j;
  A[3] += fj * j;
  b[0]  += fi * delta;
  b[1]  += fj * delta;
}

/**
 * \brief Compute quantized normal image from depth image.
 *
 * Implements section 2.6 "Extension to Dense Depth Sensors."
 *
 * \param[in]  src  The source 16-bit depth image (in mm).
 * \param[out] dst  The destination 8-bit image. Each bit represents one bin of
 *                  the view cone.
 * \param distance_threshold   Ignore pixels beyond this distance.
 * \param difference_threshold When computing normals, ignore contributions of pixels whose
 *                             depth difference with the central pixel is above this threshold.
 *
 * \todo Should also need camera model, or at least focal lengths? Replace distance_threshold with mask?
 */
void quantizedNormals(const Mat& src, Mat& dst, int distance_threshold,
                      int difference_threshold)
{
  dst = Mat::zeros(src.size(), CV_8U);

  IplImage src_ipl = src;
  IplImage* ap_depth_data = &src_ipl;
  IplImage dst_ipl = dst;
  IplImage* dst_ipl_ptr = &dst_ipl;
  IplImage** m_dep = &dst_ipl_ptr;

  unsigned short * lp_depth   = (unsigned short *)ap_depth_data->imageData;
  unsigned char  * lp_normals = (unsigned char *)m_dep[0]->imageData;

  const int l_W = ap_depth_data->width;
  const int l_H = ap_depth_data->height;

  const int l_r = 5; // used to be 7
  const int l_offset0 = -l_r - l_r * l_W;
  const int l_offset1 =    0 - l_r * l_W;
  const int l_offset2 = +l_r - l_r * l_W;
  const int l_offset3 = -l_r;
  const int l_offset4 = +l_r;
  const int l_offset5 = -l_r + l_r * l_W;
  const int l_offset6 =    0 + l_r * l_W;
  const int l_offset7 = +l_r + l_r * l_W;

  const int l_offsetx = GRANULARITY / 2;
  const int l_offsety = GRANULARITY / 2;

  for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
  {
    unsigned short * lp_line = lp_depth + (l_y * l_W + l_r);
    unsigned char * lp_norm = lp_normals + (l_y * l_W + l_r);

    for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
    {
      long l_d = lp_line[0];

      if (l_d < distance_threshold)
      {
        // accum
        long l_A[4]; l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
        long l_b[2]; l_b[0] = l_b[1] = 0;
        accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset1] - l_d,    0, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset3] - l_d, -l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset4] - l_d, +l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset6] - l_d,    0, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

        // solve
        long l_det =  l_A[0] * l_A[3] - l_A[1] * l_A[1];
        long l_ddx =  l_A[3] * l_b[0] - l_A[1] * l_b[1];
        long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

        /// @todo Magic number 1150 is focal length? This is something like
        /// f in SXGA mode, but in VGA is more like 530.
        float l_nx = static_cast<float>(1150 * l_ddx);
        float l_ny = static_cast<float>(1150 * l_ddy);
        float l_nz = static_cast<float>(-l_det * l_d);

        float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

        if (l_sqrt > 0)
        {
          float l_norminv = 1.0f / (l_sqrt);

          l_nx *= l_norminv;
          l_ny *= l_norminv;
          l_nz *= l_norminv;

          //*lp_norm = fabs(l_nz)*255;

          int l_val1 = static_cast<int>(l_nx * l_offsetx + l_offsetx);
          int l_val2 = static_cast<int>(l_ny * l_offsety + l_offsety);
          int l_val3 = static_cast<int>(l_nz * GRANULARITY + GRANULARITY);

          *lp_norm = NORMAL_LUT[l_val3][l_val2][l_val1];
        }
        else
        {
          *lp_norm = 0; // Discard shadows from depth sensor
        }
      }
      else
      {
        *lp_norm = 0; //out of depth
      }
      ++lp_line;
      ++lp_norm;
    }
  }
  cvSmooth(m_dep[0], m_dep[0], CV_MEDIAN, 5, 5);
}

class DepthNormalPyramid : public QuantizedPyramid
{
public:
  DepthNormalPyramid(const Mat& src, const Mat& mask,
                     int distance_threshold, int difference_threshold, size_t num_features,
                     int extract_threshold);

  virtual void quantize(Mat& dst) const;

  virtual void quantizeRGB(Mat& dst) const;


  virtual bool extractTemplate(Template& templ) const;

  virtual void pyrDown();

protected:
  Mat mask;

  int pyramid_level;
  Mat normal;

  size_t num_features;
  int extract_threshold;
};

DepthNormalPyramid::DepthNormalPyramid(const Mat& src, const Mat& mask,
                                       int distance_threshold, int difference_threshold, size_t num_features,
                                       int extract_threshold)
  : mask(mask),
    pyramid_level(0),
    num_features(num_features),
    extract_threshold(extract_threshold)
{
  quantizedNormals(src, normal, distance_threshold, difference_threshold);
}

void DepthNormalPyramid::pyrDown()
{
  // Some parameters need to be adjusted
  num_features /= 2; /// @todo Why not 4?
  extract_threshold /= 2;
  ++pyramid_level;

  // In this case, NN-downsample the quantized image
  Mat next_normal;
  Size size(normal.cols / 2, normal.rows / 2);
  resize(normal, next_normal, size, 0.0, 0.0, CV_INTER_NN);
  normal = next_normal;
  if (!mask.empty())
  {
    Mat next_mask;
    resize(mask, next_mask, size, 0.0, 0.0, CV_INTER_NN);
    mask = next_mask;
  }
}

void DepthNormalPyramid::quantize(Mat& dst) const
{
  dst = Mat::zeros(normal.size(), CV_8U);
  normal.copyTo(dst, mask);
}

void DepthNormalPyramid::quantizeRGB(Mat& dst) const
{
  dst = Mat::zeros(normal.size(), CV_8U);
  normal.copyTo(dst, mask);
}

bool DepthNormalPyramid::extractTemplate(Template& templ) const
{
  // Features right on the object border are unreliable
  Mat local_mask;
  if (!mask.empty())
  {
    erode(mask, local_mask, Mat(), Point(-1,-1), 2, BORDER_REPLICATE);
  }
  
  // Compute distance transform for each individual quantized orientation
  Mat temp = Mat::zeros(normal.size(), CV_8U);
  Mat distances[8];
  for (int i = 0; i < 8; ++i)
  {
    temp.setTo(1 << i, local_mask);
    bitwise_and(temp, normal, temp);
    // temp is now non-zero at pixels in the mask with quantized orientation i
    distanceTransform(temp, distances[i], CV_DIST_C, 3);
  }

  // Count how many features taken for each label
  int label_counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create sorted list of candidate features
  std::vector<Candidate> candidates;
  bool no_mask = local_mask.empty();
  for (int r = 0; r < normal.rows; ++r)
  {
    const uchar* normal_r = normal.ptr<uchar>(r);
    const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

    for (int c = 0; c < normal.cols; ++c)
    {
      if (no_mask || mask_r[c])
      {
        uchar quantized = normal_r[c];

        if (quantized != 0 && quantized != 255) // background and shadow
        {
          int label = getLabel(quantized);

          // Accept if distance to a pixel belonging to a different label is greater than
          // some threshold. IOW, ideal feature is in the center of a large homogeneous
          // region.
          float score = distances[label].at<float>(r, c);
          if (score >= extract_threshold)
          {
            candidates.push_back( Candidate(c, r, label, score) );
            ++label_counts[label];
          }
        }
      }
    }
  }
  // We require a certain number of features
  if (candidates.size() < num_features)
    return false;

  // Prefer large distances, but also want to collect features over all 8 labels.
  // So penalize labels with lots of candidates.
  for (size_t i = 0; i < candidates.size(); ++i)
  {
    Candidate& c = candidates[i];
    c.score /= (float)label_counts[c.f.label];
  }
  std::stable_sort(candidates.begin(), candidates.end());

  // Use heuristic based on object area for initial distance threshold
  int area = static_cast<int>(no_mask ? normal.total() : countNonZero(local_mask));
  float distance = sqrtf(static_cast<float>(area)) / sqrtf(static_cast<float>(num_features)) + 1.5f;
  selectScatteredFeatures(candidates, templ.features, num_features, distance);

  // Size determined externally, needs to match templates for other modalities
  templ.width = -1;
  templ.height = -1;
  templ.pyramid_level = pyramid_level;

  return true;
}

DepthNormal::DepthNormal()
  : distance_threshold(2000),
    difference_threshold(50),
    num_features(63),
    extract_threshold(2)
{
}

DepthNormal::DepthNormal(int distance_threshold, int difference_threshold, size_t num_features,
                         int extract_threshold)
  : distance_threshold(distance_threshold),
    difference_threshold(difference_threshold),
    num_features(num_features),
    extract_threshold(extract_threshold)
{
}

static const char DN_NAME[] = "DepthNormal";

std::string DepthNormal::name() const
{
  return DN_NAME;
}

Ptr<QuantizedPyramid> DepthNormal::processImpl(const Mat& src,
                                                   const Mat& mask) const
{
  return new DepthNormalPyramid(src, mask, distance_threshold, difference_threshold,
                                num_features, extract_threshold);
}

void DepthNormal::read(const FileNode& fn)
{
  std::string type = fn["type"];
  CV_Assert(type == DN_NAME);

  distance_threshold = fn["distance_threshold"];
  difference_threshold = fn["difference_threshold"];
  num_features = int(fn["num_features"]);
  extract_threshold = fn["extract_threshold"];
}

void DepthNormal::write(FileStorage& fs) const
{
  fs << "type" << DN_NAME;
  fs << "distance_threshold" << distance_threshold;
  fs << "difference_threshold" << difference_threshold;
  fs << "num_features" << int(num_features);
  fs << "extract_threshold" << extract_threshold;
}

/****************************************************************************************\
*                                 Response maps                                          *
\****************************************************************************************/

void orUnaligned8u(const uchar * src, const int src_stride,
                   uchar * dst, const int dst_stride,
                   const int width, const int height)
{
#if CV_SSE2
  volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
  volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
//haveSSE2 = false;
//haveSSE3 = false;
  bool src_aligned = reinterpret_cast<unsigned long long>(src) % 16 == 0;
#endif

	for (int r = 0; r < height; ++r)
	{
	    int c = 0;

	    #if CV_SSE2
		// Use aligned loads if possible
		if (haveSSE2 && src_aligned)
		{
		  for ( ; c < width - 15; c += 16)
		  {
		    const __m128i* src_ptr = reinterpret_cast<const __m128i*>(src + c);
		    __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
		    *dst_ptr = _mm_or_si128(*dst_ptr, *src_ptr);
		  }
		}
	    #if CV_SSE3
		// Use LDDQU for fast unaligned load
		else if (haveSSE3)
		{
		  for ( ; c < width - 15; c += 16)
		  {
		    __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src + c));
		    __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
		    *dst_ptr = _mm_or_si128(*dst_ptr, val);
		  }
		}
	    #endif
		// Fall back to MOVDQU
		else if (haveSSE2)
		{
		  for ( ; c < width - 15; c += 16)
		  {
		    __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
		    __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
		    *dst_ptr = _mm_or_si128(*dst_ptr, val);
		  }
		}    
	    #endif

		for ( ; c < width; ++c)
		  dst[c] |= src[c];

		// Advance to next row
		src += src_stride;
		dst += dst_stride;
	}
    
  
}

/**
 * \brief Spread binary labels in a quantized image.
 *
 * Implements section 2.3 "Spreading the Orientations."
 *
 * \param[in]  src The source 8-bit quantized image.
 * \param[out] dst Destination 8-bit spread image.
 * \param      T   Sampling step. Spread labels T/2 pixels in each direction.
 */
void spread(const Mat& src, Mat& dst, int T)
{
  // Allocate and zero-initialize spread (OR'ed) image
  dst = Mat::zeros(src.size(), CV_8U);

  // Fill in spread gradient image (section 2.3)
  for (int r = 0; r < T; ++r)
  {
    int height = src.rows - r;
    for (int c = 0; c < T; ++c)
    {
      orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
                    static_cast<const int>(dst.step1()), src.cols - c, height);
    }
  }
}

// Auto-generated by create_similarity_lut.py
CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
//static const unsigned char SIMILARITY_RGB_LUT[8][256] = {{0, 4, 0, 4, 0, 4, 0, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 0, 4, 0, 4, 0, 4, 0, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4}, {0, 0, 4, 4, 0, 0, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4}, {0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4}, {0, 3, 3, 3, 0, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 3, 3, 3, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 3, 3, 3, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 3, 3, 3, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 3, 0, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 3, 1, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}};
static const unsigned char SIMILARITY_RGB_LUT[8][256] = {{0, 4, 0, 4, 0, 4, 0, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 0, 4, 0, 4, 0, 4, 0, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1, 4, 1, 4, 1, 4, 1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1, 4, 1, 4, 1, 4, 1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1, 4, 1, 4, 1, 4, 1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1, 4, 1, 4, 1, 4, 1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1, 4, 1, 4, 1, 4, 1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1, 4, 1, 4, 1, 4, 1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4}, {0, 0, 4, 4, 0, 0, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4}, {0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4}, {0, 2, 2, 2, 0, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 2, 2, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 2, 2, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 2, 2, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 2, 0, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}};
/**
 * \brief Precompute response maps for a spread quantized image.
 *
 * Implements section 2.4 "Precomputing Response Maps."
 *
 * \param[in]  src           The source 8-bit spread quantized image.
 * \param[out] response_maps Vector of 8 response maps, one for each bit label.
 */
void computeResponseMaps(const Mat& src, std::vector<Mat>& response_maps)
{

  CV_Assert((src.rows * src.cols) % 16 == 0);

  // Allocate response maps
  response_maps.resize(8);
  for (int i = 0; i < 8; ++i)
    response_maps[i].create(src.size(), CV_8U);
  
  Mat lsb4(src.size(), CV_8U);
  Mat msb4(src.size(), CV_8U);
  
  for (int r = 0; r < src.rows; ++r)
  {
    const uchar* src_r = src.ptr(r);
    uchar* lsb4_r = lsb4.ptr(r);
    uchar* msb4_r = msb4.ptr(r);
    
    for (int c = 0; c < src.cols; ++c)
    {
      // Least significant 4 bits of spread image pixel
      lsb4_r[c] = src_r[c] & 15;
      // Most significant 4 bits, right-shifted to be in [0, 16)
      msb4_r[c] = (src_r[c] & 240) >> 4;
      /*if((int)src_r[c] > 0)
		if(DEBUGGING) std::cout<<"src_r: "<<(int)src_r[c]<<" - lsb4: "<<(int)lsb4_r[c]<< " - msb4: "<< (int)msb4_r[c]<<std::endl;
		*/
    }
  }
/*
#if CV_SSSE3
  volatile bool haveSSSE3 = checkHardwareSupport(CV_CPU_SSSE3);
  if (haveSSSE3)
  {
    const __m128i* lut = reinterpret_cast<const __m128i*>(SIMILARITY_LUT);
    for (int ori = 0; ori < 8; ++ori)
    {
      __m128i* map_data = response_maps[ori].ptr<__m128i>();
      __m128i* lsb4_data = lsb4.ptr<__m128i>();
      __m128i* msb4_data = msb4.ptr<__m128i>();

      // Precompute the 2D response map S_i (section 2.4)
      for (int i = 0; i < (src.rows * src.cols) / 16; ++i)
      {
        // Using SSE shuffle for table lookup on 4 orientations at a time
        // The most/least significant 4 bits are used as the LUT index
        __m128i res1 = _mm_shuffle_epi8(lut[2*ori + 0], lsb4_data[i]);
        __m128i res2 = _mm_shuffle_epi8(lut[2*ori + 1], msb4_data[i]);

        // Combine the results into a single similarity score
        map_data[i] = _mm_max_epu8(res1, res2);
      }
      //if(DEBUGGING) std::cout<<"responseMap[0] in sss3: "<<response_maps[0]<<std::endl;
    }
    
    
	
  }
  else
#endif
*/
  {
    // For each of the 8 quantized orientations...
    for (int ori = 0; ori < 8; ++ori)
    {
      uchar* map_data = response_maps[ori].ptr<uchar>();
      uchar* lsb4_data = lsb4.ptr<uchar>();
      uchar* msb4_data = msb4.ptr<uchar>();
      const uchar* lut_low = SIMILARITY_LUT + 32*ori;
      const uchar* lut_hi = lut_low + 16;

      for (int i = 0; i < src.rows * src.cols; ++i)
      {
        map_data[i] = std::max(lut_low[ lsb4_data[i] ], lut_hi[ msb4_data[i] ]);
        //if(DEBUGGING) std::cout<<"mapdata["<<i<<"]: "<<(int)map_data[i]<<std::endl;
      }
      
    }
  }
}


void computeResponseMapsRGB(const Mat& src, std::vector<Mat>& response_maps)
{
	//cambiaRGB
	uchar tableRGB[8][8];
	
	//nuovo punteggio
	//uchar tableRGB16[8][8] = {{16,0,0,4,4,0,1,1}, {0,16,0,4,0,4,1,1}, {0,0,16,0,4,4,1,1}, {4,4,0,16,2,2,4,4}, {4,0,4,2,16,2,4,4}, {0,4,4,2,2,16,4,4}, {1,1,1,2,2,2,16,0}, {1,1,1,2,2,2,0,16}};
	
	uchar tableRGB16[8][8] = {{16,0,0,2,2,0,1,1}, {0,16,0,2,0,2,1,1}, {0,0,16,0,2,2,1,1}, {4,4,0,16,2,2,4,4}, {4,0,4,2,16,2,4,4}, {0,4,4,2,2,16,4,4}, {1,1,1,2,2,2,16,0}, {1,1,1,2,2,2,0,16}};
	
	//uchar tableRGB[8][8] = {{16,0,0,8,8,0,4,6}, {0,16,0,8,0,8,4,6}, {0,0,16,0,8,8,4,6}, {8,8,0,16,4,4,6,8}, {8,0,8,4,16,4,6,8}, {0,8,8,4,4,16,6,8}, {2,2,2,4,4,4,16,0}, {2,2,2,4,4,4,0,16}};
	
	//max
	//uchar tableRGB[8][8] = {{16,16,16,16,16,16,16,16}, {16,16,16,16,16,16,16,16}, {16,16,16,16,16,16,16,16}, {16,16,16,16,16,16,16,16}, {16,16,16,16,16,16,16,16}, {16,16,16,16,16,16,16,16}, {16,16,16,16,16,16,16,16}, {16,16,16,16,16,16,16,16}};
	//solo match uguali
	//uchar tableRGB[8][8] = {{16,0,0,0,0,0,0,0}, {0,16,0,0,0,0,0,0}, {0,0,16,0,0,0,0,0}, {0,0,0,16,0,0,0,0}, {0,0,0,0,16,0,0,0}, {0,0,0,0,0,16,0,0}, {0,0,0,0,0,0,16,0}, {0,0,0,0,0,0,0,16}};
	
	//main 
	//uchar tableRGB8[8][8] = {{4,0,0,3,3,0,2,2}, {0,4,0,3,0,3,2,2}, {0,0,4,0,3,3,2,2}, {3,3,0,4,1,1,2,2}, {3,0,3,1,4,1,2,2}, {0,3,3,1,1,4,2,2}, {1,1,1,2,2,2,4,0}, {1,1,1,2,2,2,0,4}};
	
	//rg, rb, gb abbassati a 1
	//uchar tableRGB8[8][8] = {{4,0,0,1,1,0,2,2}, {0,4,0,1,0,1,2,2}, {0,0,4,0,1,1,2,2}, {2,2,0,4,1,1,2,2}, {2,0,2,1,4,1,2,2}, {0,2,2,1,1,4,2,2}, {1,1,1,1,1,1,4,0}, {1,1,1,1,1,1,0,4}};
	
	
	//abbassati
	uchar tableRGB8[8][8] = {{4,0,0,2,2,0,1,1}, {0,4,0,2,0,2,1,1}, {0,0,4,0,2,2,1,1}, {2,2,0,4,1,1,1,1}, {2,0,2,1,4,1,1,1}, {0,2,2,1,1,4,1,1}, {1,1,1,2,2,2,4,0}, {1,1,1,2,2,2,0,4}};
	//abbassati e nero e bianco penalizzati
	//uchar tableRGB[8][8] = {{4,0,0,2,2,0,1,1}, {0,4,0,2,0,2,1,1}, {0,0,4,0,2,2,1,1}, {2,2,0,4,1,1,1,1}, {2,0,2,1,4,1,1,1}, {0,2,2,1,1,4,1,1}, {0,0,0,1,1,1,4,0}, {0,0,0,1,1,1,0,4}};
	//abbassati e nero e bianco penalizzatiSSIMI
	//uchar tableRGB[8][8] = {{4,0,0,2,2,0,1,1}, {0,4,0,2,0,2,1,1}, {0,0,4,0,2,2,1,1}, {2,2,0,4,1,1,1,1}, {2,0,2,1,4,1,1,1}, {0,2,2,1,1,4,1,1}, {0,0,0,0,0,0,4,0}, {0,0,0,0,0,0,0,4}};
	
	//solo match uguali
	//uchar tableRGB[8][8] = {{4,0,0,0,0,0,0,0}, {0,4,0,0,0,0,0,0}, {0,0,4,0,0,0,0,0}, {0,0,0,4,0,0,0,0}, {0,0,0,0,4,0,0,0}, {0,0,0,0,0,4,0,0}, {0,0,0,0,0,0,4,0}, {0,0,0,0,0,0,0,4}};
	
	//uchar tableRGB[8][8] = {{4,0,0,3,3,0,0,0}, {0,4,0,3,0,3,0,0}, {0,0,4,0,3,3,0,0}, {3,3,0,4,1,1,0,0}, {3,0,3,1,4,1,0,0}, {0,3,3,1,1,4,0,0}, {1,1,1,2,2,2,0,0}, {1,1,1,2,2,2,0,0}};
	//uchar tableRGB[8][8] = {{4,0,0,3,3,0,0,0}, {0,4,0,3,0,3,0,0}, {0,0,4,0,3,3,0,0}, {3,3,0,4,1,1,0,0}, {3,0,3,1,4,1,0,0}, {0,3,3,1,1,4,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}};
	//uchar tableRGB[8][8] = {{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0}};
	//uchar tableRGB[8][8] = {{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1}};
	//uchar tableRGB[8][8] = {{4,4,4,4,4,4,4,4},{4,4,4,4,4,4,4,4},{4,4,4,4,4,4,4,4},{4,4,4,4,4,4,4,4},{4,4,4,4,4,4,4,4},{4,4,4,4,4,4,4,4},{4,4,4,4,4,4,4,4},{4,4,4,4,4,4,4,4}};
	

	if(punteggio16 == true)
	{
	    for(int a = 0; a<8; a++)
		for(int b = 0; b<8; b++)
		    tableRGB[a][b] = tableRGB16[a][b];
	}
	else
	{
	    for(int a = 0; a<8; a++)
		for(int b = 0; b<8; b++)
		    tableRGB[a][b] = tableRGB8[a][b];
	}

  // Allocate response maps
  response_maps.resize(8);
  for (int i = 0; i < 8; ++i)
    response_maps[i].create(src.size(), CV_8UC1);
  
  //b1 meno significativo, b8 più significativo
  uchar b1,b2,b3,b4,b5,b6,b7,b8;
  
  
  int ind = 0;
  for (int r = 0; r < src.rows; ++r)
  {
    const uchar* src_r = src.ptr(r);
    
    for (int c = 0; c < src.cols; ++c)
    {
     
     uchar tmpUchar = src_r[c];
      //controllo sul grigio gray
      if(tmpUchar == 0)
      {
	  
	    for (int ori = 0; ori < 8; ori++)
	    {
		if(ori == 6 || ori == 7)
		{
		    if(punteggio16 == true)
			response_maps[ori].at<uchar>(r,c) = 8; //valore forfettario per matchare bianco e nero con grigio
		    else
			response_maps[ori].at<uchar>(r,c) = 2; //valore forfettario per matchare bianco e nero con grigio
		}
		else
		    response_maps[ori].at<uchar>(r,c) = 0;
	    }
      }
      else
      {
	  b1 = tmpUchar & 1;
	  b2 = tmpUchar & 2;
	  b3 = tmpUchar & 4;
	  b4 = tmpUchar & 8;
	  b5 = tmpUchar & 16;
	  b6 = tmpUchar & 32;
	  b7 = tmpUchar & 64;
	  b8 = tmpUchar & 128;
	  
	  for (int ori = 0; ori < 8; ori++)
	  {
		 uchar best_ori_value = 0;  
		     if((int)b1 != 0)
		     {
			    int tmpValue = getLabel((int)b1);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     if((int)b2 != 0)
		     {
			    int tmpValue = getLabel((int)b2);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     if((int)b3 != 0)
		     {
			    int tmpValue = getLabel((int)b3);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     if((int)b4 != 0)
		     {
			    int tmpValue = getLabel((int)b4);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     if((int)b5 != 0)
		     {
			    int tmpValue = getLabel((int)b5);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     if((int)b6 != 0)
		     {
			    int tmpValue = getLabel((int)b6);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     if((int)b7!= 0)
		     {
			    int tmpValue = getLabel((int)b7);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     if((int)b8 != 0)
		     {
			    int tmpValue = getLabel((int)b8);
			    if(tableRGB[tmpValue][ori] > best_ori_value)
				    best_ori_value = tableRGB[tmpValue][ori];
		     }
		     
		     //std::cout<<"ori_value: "<<(int)best_ori_value<<std::endl;
		     response_maps[ori].at<uchar>(r,c) = best_ori_value;
	      }
      }
      
		
		
		
    }
    
  }

}

/*
void computeResponseMapsRGB(const Mat& src, std::vector<Mat>& response_maps)
{
	
  // Allocate response maps
  response_maps.resize(8);
  for (int i = 0; i < 8; ++i)
    response_maps[i].create(src.size(), CV_8UC1);
  
  
  int ind = 0;
  for (int r = 0; r < src.rows; ++r)
  {
    const uchar* src_r = src.ptr(r);
    
    for (int c = 0; c < src.cols; ++c)
    {
      
      for (int ori = 0; ori < 8; ori++)
      {
	    response_maps[ori].at<uchar>(r,c) = SIMILARITY_RGB_LUT[ori][(int)src_r[c]];
      }
		
		
		
    }
    
  }

}
*/

/**
 * \brief Convert a response map to fast linearized ordering.
 *
 * Implements section 2.5 "Linearizing the Memory for Parallelization."
 *
 * \param[in]  response_map The 2D response map, an 8-bit image.
 * \param[out] linearized   The response map in linearized order. It has T*T rows,
 *                          each of which is a linear memory of length (W/T)*(H/T).
 * \param      T            Sampling step.
 */
void linearize(const Mat& response_map, Mat& linearized, int T)
{
//if(DEBUGGING) std::cout<<"size in LINEARIZE: "<<"rows: "<<response_map.rows<< " - cols: "<<response_map.cols<< " - T: "<<T<<std::endl;

  CV_Assert(response_map.rows % T == 0);
  CV_Assert(response_map.cols % T == 0);

  // linearized has T^2 rows, where each row is a linear memory
  int mem_width = response_map.cols / T;
  int mem_height = response_map.rows / T;
  
  linearized.create(T*T, mem_width * mem_height, CV_8U);

  
  // Outer two for loops iterate over top-left T^2 starting pixels
  int index = 0;
  for (int r_start = 0; r_start < T; ++r_start)
  {
    for (int c_start = 0; c_start < T; ++c_start)
    {
      uchar* memory_uchar = linearized.ptr(index);
      ++index;
      
      // Inner two loops copy every T-th pixel into the linear memory
      for (int r = r_start; r < response_map.rows; r += T)
      {
        const uchar* response_data = response_map.ptr(r);
	
        for (int c = c_start; c < response_map.cols; c += T)        
		*memory_uchar++ = response_data[c];
      }
    }
  }
  
  /*imshow("response map in linearize", response_map);
  imshow("response map inlinearize LINEARIZZATA", linearized);
  waitKey(0);*/
}

/****************************************************************************************\
*                               Linearized similarities                                  *
\****************************************************************************************/

const unsigned char* accessLinearMemory(const std::vector<Mat>& linear_memories,
					const Feature& f, int T, int W)
{
  // Retrieve the TxT grid of linear memories associated with the feature label
  const Mat& memory_grid = linear_memories[f.label];
  CV_DbgAssert(memory_grid.rows == T*T);
  CV_DbgAssert(f.x >= 0);
  CV_DbgAssert(f.y >= 0);
  // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
  int grid_x = f.x % T;
  int grid_y = f.y % T;
  int grid_index = grid_y * T + grid_x;
  CV_DbgAssert(grid_index >= 0);
  CV_DbgAssert(grid_index < memory_grid.rows);
  const unsigned char* memory = memory_grid.ptr(grid_index);
  // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
  // input image width decimated by T.
  int lm_x = f.x / T;
  int lm_y = f.y / T;
  int lm_index = lm_y * W + lm_x;
  CV_DbgAssert(lm_index >= 0);
  CV_DbgAssert(lm_index < memory_grid.cols);
  return memory + lm_index;
}


const unsigned char* accessLinearMemoryRGB(const std::vector<Mat>& linear_memories,
					const Feature& f, int T, int W)
{
  // Retrieve the TxT grid of linear memories associated with the feature label
  const Mat& memory_grid = linear_memories[f.rgbLabel];
  //if(DEBUGGING) std::cout<<"rgbLabel: "<<f.rgbLabel<<std::endl;
  
  /*
  if(DEBUGGING) std::cout<<"rgbLabel: "<<f.rgbLabel<<std::endl;
  imshow("memory_grid", memory_grid);
  if(DEBUGGING) std::cout<<"sizeof linear_memories: "<<linear_memories.size()<<std::endl;
  
  imshow("linear memories in similarity 0", linear_memories[0]);
  imshow("linear memories in similarity 1", linear_memories[1]);
  imshow("linear memories in similarity 2", linear_memories[2]);
  imshow("linear memories in similarity 3", linear_memories[3]);
  imshow("linear memories in similarity 4", linear_memories[4]);
  imshow("linear memories in similarity 5", linear_memories[5]);
  imshow("linear memories in similarity 6", linear_memories[6]);
  imshow("linear memories in similarity 7", linear_memories[7]);
  waitKey(0);
  */
  
  CV_DbgAssert(memory_grid.rows == T*T);
  CV_DbgAssert(f.x >= 0);
  CV_DbgAssert(f.y >= 0);
  // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
  int grid_x = f.x % T;
  int grid_y = f.y % T;
  int grid_index = grid_y * T + grid_x;
  CV_DbgAssert(grid_index >= 0);
  CV_DbgAssert(grid_index < memory_grid.rows);
  const unsigned char* memory = memory_grid.ptr(grid_index);
  // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
  // input image width decimated by T.
  int lm_x = f.x / T;
  int lm_y = f.y / T;
  int lm_index = lm_y * W + lm_x;
  CV_DbgAssert(lm_index >= 0);
  CV_DbgAssert(lm_index < memory_grid.cols);
  return memory + lm_index;
}

/**
 * \brief Compute similarity measure for a given template at each sampled image location.
 *
 * Uses linear memories to compute the similarity measure as described in Fig. 7.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image of size (W/T, H/T).
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 */
void similarity(const std::vector<Mat>& linear_memories, const Template& templ,
                Mat& dst, Size size, int T)
{
  // 63 features or less is a special case because the max similarity per-feature is 4.
  // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
  // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
  // general function would use _mm_add_epi16.
  CV_Assert(templ.features.size() <= 63);
  /// @todo Handle more than 255/MAX_RESPONSE features!!

  // Decimate input image size by factor of T
  int W = size.width / T;
  int H = size.height / T;

  // Feature dimensions, decimated by factor T and rounded up
  int wf = (templ.width - 1) / T + 1;
  int hf = (templ.height - 1) / T + 1;

  // Span is the range over which we can shift the template around the input image
  int span_x = W - wf;
  int span_y = H - hf;

  // Compute number of contiguous (in memory) pixels to check when sliding feature over
  // image. This allows template to wrap around left/right border incorrectly, so any
  // wrapped template matches must be filtered out!
  int template_positions = span_y * W + span_x + 1; // why add 1?
  //int template_positions = (span_y - 1) * W + span_x; // More correct?

  /// @todo In old code, dst is buffer of size m_U. Could make it something like
  /// (span_x)x(span_y) instead?
  dst = Mat::zeros(H, W, CV_8U);
  uchar* dst_ptr = dst.ptr<uchar>();

#if CV_SSE2
  volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
  volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif

//haveSSE2 = false;
//haveSSE3 = false;

  // Compute the similarity measure for this template by accumulating the contribution of
  // each feature
  for (int i = 0; i < (int)templ.features.size(); ++i)
  {
    // Add the linear memory at the appropriate offset computed from the location of
    // the feature in the template
    Feature f = templ.features[i];
    // Discard feature if out of bounds
    /// @todo Shouldn't actually see x or y < 0 here?
    if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
      continue;
    const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

    

    // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
    int j = 0;
    // Process responses 16 at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
    if (haveSSE3)
    {
      // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
      for ( ; j < template_positions - 15; j += 16)
      {
        __m128i responses = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
        __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
        *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
      }
    }
    else
#endif
    if (haveSSE2)
    {
      // Fall back to MOVDQU
      for ( ; j < template_positions - 15; j += 16)
      {
        __m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
        __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
        *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
      }
    }
#endif

    for ( ; j < template_positions; ++j)
      dst_ptr[j] += lm_ptr[j];
  }
}

/**
 * \brief Compute similarity measure for a given template at each sampled image location.
 *
 * Uses linear memories to compute the similarity measure as described in Fig. 7.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image of size (W/T, H/T).
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 */
void similarityRGB(const std::vector<Mat>& linear_memories, const std::vector<Mat>& linear_memories_rgb, const Template& templ,
                Mat& dst, Mat& dst_rgb, Size size, int T)
{
    



  //if(DEBUGGING) std::cout<<"similarity debug: linear_memories.size(): "<<linear_memories.size()<<" righe di linear_memories: "<<linear_memories[0].rows<<" colonne: "<<linear_memories[0].cols<<" Size.width: "<<size.width<<" Size.height: "<<size.height<<" T: "<<T<<" template.width: "<<templ.width<<" template.height: "<<templ.height<<"pyramid_level: "<<templ.pyramid_level<<std::endl;
	
	
  // 63 features or less is a special case because the max similarity per-feature is 4.
  // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
  // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
  // general function would use _mm_add_epi16.
  
  
  //CV_Assert(templ.features.size() <= 63);
  
  //if(DEBUGGING) std::cout<<"size.width: "<<size.width<<" T: "<<T<<std::endl;
  
  /// @todo Handle more than 255/MAX_RESPONSE features!!

  // Decimate input image size by factor of T
  int W = size.width / T;
  int H = size.height / T;

  // Feature dimensions, decimated by factor T and rounded up
  int wf = (templ.width - 1) / T + 1;
  int hf = (templ.height - 1) / T + 1;

  // Span is the range over which we can shift the template around the input image
  int span_x = W - wf;
  int span_y = H - hf;

  // Compute number of contiguous (in memory) pixels to check when sliding feature over
  // image. This allows template to wrap around left/right border incorrectly, so any
  // wrapped template matches must be filtered out!
  int template_positions = span_y * W + span_x + 1; // why add 1?
  //int template_positions = (span_y - 1) * W + span_x; // More correct?

  /// @todo In old code, dst is buffer of size m_U. Could make it something like
  /// (span_x)x(span_y) instead?
  if(use63 == true && punteggio16 == false && featuresSignatureCandidates == false)
  {
	dst = Mat::zeros(H, W, CV_8U);
	dst_rgb = Mat::zeros(H, W, CV_8U);
  }
  else
  { 
	dst = Mat::zeros(H, W, CV_16U);
	dst_rgb = Mat::zeros(H, W, CV_16U);
  }
  //ptr to dst matrix if we are using max 63 features
  uchar* dst_ptr_uchar = dst.ptr<uchar>();
  //ptr to dst matrix if we are using more than 63 features
  ushort* dst_ptr_ushort = dst.ptr<ushort>();
  //ptr to dst_rgb matrix if we are using max 63 features
  uchar* dst_ptr_rgb_uchar = dst_rgb.ptr<uchar>();
  //ptr to dst_rgb matrix if we are using more than 63 features
  ushort* dst_ptr_rgb_ushort = dst_rgb.ptr<ushort>();

//#if CV_SSE2
  volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
  volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
//#endif
//haveSSE2 = false;
//haveSSE3 = false;
  // Compute the similarity measure for this template by accumulating the contribution of
  // each feature
Timer extract_timer;
	extract_timer.start();

  int total_features_size;
  std::vector<cv::my_linemod::Feature> total_features;
  
  total_features_size = (int)templ.features.size();
  total_features.insert(total_features.end(), templ.features.begin(), templ.features.end());
  
  //offset che indica se stiamo ancora valutando le feature normali o solo quelle della signature, in tal caso dobbiamo solo svolgere la parte RGB
  int offsetEndNormalFeatures = total_features_size;
  
  if(featuresSignatureCandidates == true)
  {
      total_features_size += (int)templ.featuresSignature.size();
      total_features.insert(total_features.end(), templ.featuresSignature.begin(), templ.featuresSignature.end());
  }
  
  for (int i = 0; i < total_features_size; ++i)
  {
    bool onlyFeatureSignature = false;
    if(i >= offsetEndNormalFeatures)
	onlyFeatureSignature = true;
	
    // Add the linear memory at the appropriate offset computed from the location of
    // the feature in the template
    Feature f = total_features[i];
    // Discard feature if out of bounds
    
    
    
    /// @todo Shouldn't actually see x or y < 0 here?
    if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
      continue;
    
    const uchar* lm_ptr;
    if(onlyFeatureSignature == false)
	lm_ptr = accessLinearMemory(linear_memories, f, T, W);
    const uchar* lm_ptr_rgb = accessLinearMemoryRGB(linear_memories_rgb, f, T, W);
   
    
    //QUI COMPARE ERRORE
    //if(DEBUGGING) std::cout<<std::endl<<"i: "<<i<<" - valore: "<<(int)*lm_ptr/*<<" - indirizzo: "<<static_cast<const void *>(lm_ptr)*/<<std::endl;
   
   
   
   
   /* if(DEBUGGING) std::cout<<"template_positions: "<<template_positions<<std::endl;
	for(int k = 0; k<template_positions; k++)
		if(DEBUGGING) std::cout<<"lm_ptr["<<k<<"]: "<<(int)lm_ptr[k]<<std::endl;
*/



    // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
    int j = 0;

if(use63 == true && punteggio16 == false && featuresSignatureCandidates == false)
{
	// Process responses 16 at a time if vectorization possible
	#if CV_SSE2
	#if CV_SSE3
    if (haveSSE3)
    {
    
      // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
      for ( ; j < template_positions - 15; j += 16)
      {
        __m128i responses = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
        __m128i responsesRGB = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb + j));
	
        __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr_uchar + j);
	__m128i* dst_ptr_rgb_sse = reinterpret_cast<__m128i*>(dst_ptr_rgb_uchar + j);
	
        *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
	*dst_ptr_rgb_sse = _mm_add_epi8(*dst_ptr_rgb_sse, responsesRGB);
      }
    }
    else
#endif
    if (haveSSE2)
    {
	
      // Fall back to MOVDQU
      for ( ; j < template_positions - 15; j += 16)
      {
        __m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
	__m128i responsesRGB = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb + j));
	
        __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr_uchar + j);
        __m128i* dst_ptr_rgb_sse = reinterpret_cast<__m128i*>(dst_ptr_rgb_uchar + j);
	
        *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
	*dst_ptr_rgb_sse = _mm_add_epi8(*dst_ptr_rgb_sse, responsesRGB);
      }
    }
#endif
}
else //if we are working with more than 63 features !!!!!!!!!!!!!!!!!!!!AGGIUNGERE RGB!!!!!!!!!!!!!!!!
{
	// Process responses 8 at a time if vectorization possible
	#if CV_SSE2
	#if CV_SSE3
    if (haveSSE3)
    {
      // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
      for ( ; j < template_positions - 15; j += 16)
      {
	
	if(onlyFeatureSignature == false)
	{
	    __m128i v_aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
	    __m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
	    __m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());
	    
	    __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr_ushort + j);
	    
	    dst_ptr_sse[0] = _mm_add_epi16(dst_ptr_sse[0], v_alignedLo);
	    dst_ptr_sse[1] = _mm_add_epi16(dst_ptr_sse[1], v_alignedHi);
	} 
	
	
	__m128i v_aligned_rgb = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb + j));
	__m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
	__m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

	__m128i* dst_ptr_rgb_sse = reinterpret_cast<__m128i*>(dst_ptr_rgb_ushort + j);
	
	dst_ptr_rgb_sse[0] = _mm_add_epi16(dst_ptr_rgb_sse[0], v_alignedLo_rgb);
	dst_ptr_rgb_sse[1] = _mm_add_epi16(dst_ptr_rgb_sse[1], v_alignedHi_rgb);
	  

      }
    }
    else
#endif
    if (haveSSE2)
    {
      // Fall back to MOVDQU
      for ( ; j < template_positions - 15; j += 16)
      {
	    if(onlyFeatureSignature == false)
	    {
		__m128i v_aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
		__m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
		__m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());
		
		__m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr_ushort + j);
		
		dst_ptr_sse[0] = _mm_add_epi16(dst_ptr_sse[0], v_alignedLo);
		dst_ptr_sse[1] = _mm_add_epi16(dst_ptr_sse[1], v_alignedHi);
	    }
	    
	    
	    __m128i v_aligned_rgb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb + j));
	    __m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
	    __m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

	    __m128i* dst_ptr_rgb_sse = reinterpret_cast<__m128i*>(dst_ptr_rgb_ushort + j);
	
	    dst_ptr_rgb_sse[0] = _mm_add_epi16(dst_ptr_rgb_sse[0], v_alignedLo_rgb);
	    dst_ptr_rgb_sse[1] = _mm_add_epi16(dst_ptr_rgb_sse[1], v_alignedHi_rgb);
        

      }
    }
#endif
}


	//if(DEBUGGING) std::cout<<"j: "<<j<<" - template_postions: "<<template_positions<<std::endl;

	if(use63 == true && punteggio16 == false && featuresSignatureCandidates == false)
	{
		for ( ; j < template_positions; ++j)
		{
			
			/*if(lm_ptr_rgb[j] > 0)
			{
				if(DEBUGGING) std::cout<<"VALORE alla posizione "<<j<<" è: "<<(int)lm_ptr_rgb[j]<<std::endl;
				waitKey(0);
			}*/
			dst_ptr_uchar[j] += lm_ptr[j];
			dst_ptr_rgb_uchar[j] += lm_ptr_rgb[j];
		}
	}
	else
	{
		for ( ; j < template_positions; ++j)
		{
		    if(onlyFeatureSignature == false)
			dst_ptr_ushort[j] += lm_ptr[j];
		    dst_ptr_rgb_ushort[j] += lm_ptr_rgb[j];
		}
		
	//	if(DEBUGGING) std::cout<<std::endl<<"i: "<<i<<" - lm_ptr[j]: "<<lm_ptr[j]<<std::endl;
	}
	
	//if(DEBUGGING) std::cout<<std::endl<<"j: "<<i<<" - lm_ptr[j]: "<<lm_ptr[j]<<std::endl;

    
  }

  extract_timer.stop();

  
 
}

/**
 * \brief Compute similarity measure for a given template in a local region.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image, 16x16.
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 * \param      center          Center of the local region.
 */
void similarityLocal(const std::vector<Mat>& linear_memories, const Template& templ,
                     Mat& dst, Size size, int T, Point center)
{
    
    if(DEBUGGING) std::cout<<"SIMILARITYLOCAL"<<std::endl;
  // Similar to whole-image similarity() above. This version takes a position 'center'
  // and computes the energy in the 16x16 patch centered on it.
  CV_Assert(templ.features.size() <= 63);

  // Compute the similarity map in a 16x16 patch around center
  int W = size.width / T;
  dst = Mat::zeros(16, 16, CV_8U);

  // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
  // center to get the top-left corner of the 16x16 patch.
  // NOTE: We make the offsets multiples of T to agree with results of the original code.
  int offset_x = (center.x / T - 8) * T;
  int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
  volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
  volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
  __m128i* dst_ptr_sse = dst.ptr<__m128i>();
#endif
//haveSSE2 = false;
//haveSSE3 = false;
  for (int i = 0; i < (int)templ.features.size(); ++i)
  {
    Feature f = templ.features[i];
    f.x += offset_x;
    f.y += offset_y;
    // Discard feature if out of bounds, possibly due to applying the offset
    if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
      continue;

    const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

    // Process whole row at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
    if (haveSSE3)
    {
      // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
      for (int row = 0; row < 16; ++row)
      {
        __m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
        dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
        lm_ptr += W; // Step to next row
      }
    }
    else
#endif
    if (haveSSE2)
    {
      // Fall back to MOVDQU
      for (int row = 0; row < 16; ++row)
      {
        __m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
        dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
        lm_ptr += W; // Step to next row
      }
    }
    else
#endif
    {
      uchar* dst_ptr = dst.ptr<uchar>();
      for (int row = 0; row < 16; ++row)
      {
        for (int col = 0; col < 16; ++col)
          dst_ptr[col] += lm_ptr[col];
        dst_ptr += 16;
        lm_ptr += W;
      }
    }
  }
}

/**
 * \brief Compute similarity measure for a given template in a local region.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image, 16x16.
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 * \param      center          Center of the local region.
 */
void similarityLocalRGB(const std::vector<Mat>& linear_memories, const std::vector<Mat>& linear_memories_rgb, const Template& templ,
                     Mat& dst, Mat& dst_rgb, Size size, int T, Point center)
{
   
    
  // Similar to whole-image similarity() above. This version takes a position 'center'
  // and computes the energy in the 16x16 patch centered on it.
  
  //CV_Assert(templ.features.size() <= 63);

  // Compute the similarity map in a 16x16 patch around center
  int W = size.width / T;

  if(use63 == true && punteggio16 == false && featuresSignatureCandidates == false)
  {
	dst = Mat::zeros(16, 16, CV_8U);
	dst_rgb = Mat::zeros(16, 16, CV_8U);
  }
  else
  {
	dst = Mat::zeros(16, 16, CV_16U);
	dst_rgb = Mat::zeros(16, 16, CV_16U);
  }

  
  // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
  // center to get the top-left corner of the 16x16 patch.
  // NOTE: We make the offsets multiples of T to agree with results of the original code.
  int offset_x = (center.x / T - 8) * T;
  int offset_y = (center.y / T - 8) * T;


//#if CV_SSE2
  volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
  volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
  //if(haveSSE2 || haveSSE3)
	__m128i* dst_ptr_sse = dst.ptr<__m128i>();
	__m128i* dst_ptr_rgb_sse = dst_rgb.ptr<__m128i>();
haveSSE2 = false;
//haveSSE3 = false;
//#endif


  int total_features_size;
  std::vector<cv::my_linemod::Feature> total_features;
  
  total_features_size = (int)templ.features.size();
  total_features.insert(total_features.end(), templ.features.begin(), templ.features.end());
  //offset che indica se stiamo ancora valutando le feature normali o solo quelle della signature, in tal caso dobbiamo solo svolgere la parte RGB
  int offsetEndNormalFeatures = total_features_size;
  
  if(featuresSignatureCandidates == true)
  {
      total_features_size += (int)templ.featuresSignature.size();
      total_features.insert(total_features.end(), templ.featuresSignature.begin(), templ.featuresSignature.end());
  }
  
  for (int i = 0; i < total_features_size; ++i)
  {
    bool onlyFeatureSignature = false;
    if(i >= offsetEndNormalFeatures)
	onlyFeatureSignature = true;
	
    // Add the linear memory at the appropriate offset computed from the location of
    // the feature in the template
    Feature f = total_features[i];
    
    f.x += offset_x;
    f.y += offset_y;
    // Discard feature if out of bounds, possibly due to applying the offset
    if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
      continue;

    const uchar* lm_ptr;
    if(onlyFeatureSignature == false)
	lm_ptr = accessLinearMemory(linear_memories, f, T, W);
    const uchar* lm_ptr_rgb = accessLinearMemoryRGB(linear_memories_rgb, f, T, W);


    // Process whole row at a time if vectorization possible

#if CV_SSE2
#if CV_SSE3
    if (haveSSE3)
    { 

	if( use63 == true && punteggio16 == false && featuresSignatureCandidates == false)
	{

		  
	      // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
	      for (int row = 0; row < 16; ++row)
	      {
		__m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
		__m128i alignedRGB = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb));
		
	
		dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
		dst_ptr_rgb_sse[row] = _mm_add_epi8(dst_ptr_rgb_sse[row], alignedRGB);

		lm_ptr += W; // Step to next row
		lm_ptr_rgb += W; // Step to next row
	      }
	} 	
	else
	{
	    
	    // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
	      for (int row = 0; row < 32; row+=2)
	      {
		if(onlyFeatureSignature == false)
		{
		    __m128i v_aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
		    __m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
		    __m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());
		    
		    dst_ptr_sse[row] = _mm_add_epi16(dst_ptr_sse[row], v_alignedLo);
		    dst_ptr_sse[row + 1] = _mm_add_epi16(dst_ptr_sse[row + 1], v_alignedHi);
		    
		    lm_ptr += W; // Step to next row
		}
		
		__m128i v_aligned_rgb = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb));
		__m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
		__m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

		dst_ptr_rgb_sse[row] = _mm_add_epi16(dst_ptr_rgb_sse[row], v_alignedLo_rgb);
		dst_ptr_rgb_sse[row + 1] = _mm_add_epi16(dst_ptr_rgb_sse[row + 1], v_alignedHi_rgb);
		
		lm_ptr_rgb += W; // Step to next row
	      }
	}
    }
    else
#endif
    if (haveSSE2)
    {

	if(use63 == true && punteggio16 == false && featuresSignatureCandidates == false)
	  {
	      // Fall back to MOVDQU
	      for (int row = 0; row < 16; ++row)
	      {
		
		__m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
		__m128i alignedRGB = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb));
		
		dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
		dst_ptr_rgb_sse[row] = _mm_add_epi8(dst_ptr_rgb_sse[row], alignedRGB);
		
		lm_ptr += W; // Step to next row
		lm_ptr_rgb += W; // Step to next row
	      }
	}
	else
	{
	    // Fall back to MOVDQU
	      for (int row = 0; row < 32; row+=2)
	      {
		if(onlyFeatureSignature == false)
		{
		    __m128i v_aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
		    __m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
		    __m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());
		    
		    dst_ptr_sse[row] = _mm_add_epi16(dst_ptr_sse[row], v_alignedLo);
		    dst_ptr_sse[row + 1] = _mm_add_epi16(dst_ptr_sse[row + 1], v_alignedHi);
		    
		    lm_ptr += W; // Step to next row
		}
				
		__m128i v_aligned_rgb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb));
		__m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
		__m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

		dst_ptr_rgb_sse[row] = _mm_add_epi16(dst_ptr_rgb_sse[row], v_alignedLo_rgb);
		dst_ptr_rgb_sse[row + 1] = _mm_add_epi16(dst_ptr_rgb_sse[row + 1], v_alignedHi_rgb);
		
		lm_ptr_rgb += W; // Step to next row
	      }
	}
    }
    else
#endif

    {

	  if(use63 == true && punteggio16 == false && featuresSignatureCandidates == false)
	  {
		  uchar* dst_ptr = dst.ptr<uchar>();
		  uchar* dst_ptr_rgb = dst_rgb.ptr<uchar>();
		  for (int row = 0; row < 16; ++row)
		  {
			for (int col = 0; col < 16; ++col)
			{
			  dst_ptr[col] += lm_ptr[col];
			  dst_ptr_rgb[col] += lm_ptr_rgb[col];
			}
			dst_ptr += 16;
			lm_ptr += W;
			dst_ptr_rgb += 16;
			lm_ptr_rgb += W;
		  }
	  }
	  else
	  {
	      if(onlyFeatureSignature == false)
	      {
		  ushort* dst_ptr = dst.ptr<ushort>();
		  for (int row = 0; row < 16; ++row)
		  {
			for (int col = 0; col < 16; ++col)
			  dst_ptr[col] += lm_ptr[col];
			  
			dst_ptr += 16;
			lm_ptr += W;
		  }
	      }
		
		  
	      ushort* dst_ptr_rgb = dst_rgb.ptr<ushort>();
	      for (int row = 0; row < 16; ++row)
	      {
		    for (int col = 0; col < 16; ++col)
		      dst_ptr_rgb[col] += lm_ptr_rgb[col];
		    
		    dst_ptr_rgb += 16;
		    lm_ptr_rgb += W;
	      }
	  }
      
    }
    
  }
  //cout<<"local esce prima parte"<<endl;

  //if(DEBUGGING) std::cout<<"dst16x16:"<<std::endl<<dst<<std::endl;
    //  if(DEBUGGING) std::cout<<"feature in local similarity: "<<countFeatureLocal<<std::endl;
}

/**
 * \brief Compute similarity measure for a given template in a local region.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image, 16x16.
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 * \param      center          Center of the local region.
 */
void similarityLocalRGB_8(const std::vector<Mat>& linear_memories, const std::vector<Mat>& linear_memories_rgb, const Template& templ,
                     Mat& dst, Mat& dst_rgb, Size size, int T, Point center)
{
  // Similar to whole-image similarity() above. This version takes a position 'center'
  // and computes the energy in the 16x16 patch centered on it.
  
  //CV_Assert(templ.features.size() <= 63);

  // Compute the similarity map in a 16x16 patch around center
  int W = size.width / T;
  if(use63 == true)
  {
	dst = Mat::zeros(8, 8, CV_8U);
	dst_rgb = Mat::zeros(8, 8, CV_8U);
  }
  else
  {
	dst = Mat::zeros(8, 8, CV_16U);
	dst_rgb = Mat::zeros(8, 8, CV_16U);
  }
  // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
  // center to get the top-left corner of the 16x16 patch.
  // NOTE: We make the offsets multiples of T to agree with results of the original code.
  int offset_x = (center.x / T - 4) * T;
  int offset_y = (center.y / T - 4) * T;



int countFeatureLocal = 0;
  for (int i = 0; i < (int)templ.features.size(); ++i)
  {
    Feature f = templ.features[i];
    f.x += offset_x;
    f.y += offset_y;
    // Discard feature if out of bounds, possibly due to applying the offset
    if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
      continue;
    countFeatureLocal++;

    const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);
    const uchar* lm_ptr_rgb = accessLinearMemoryRGB(linear_memories_rgb, f, T, W);

    {
	  if(use63 == true)
	  {
		  uchar* dst_ptr = dst.ptr<uchar>();
		  uchar* dst_ptr_rgb = dst_rgb.ptr<uchar>();
		  for (int row = 0; row < 8; ++row)
		  {
			for (int col = 0; col < 8; ++col)
			{
			  dst_ptr[col] += lm_ptr[col];
			  dst_ptr_rgb[col] += lm_ptr_rgb[col];
		    }
			dst_ptr += 8;
			lm_ptr += W;
			dst_ptr_rgb += 8;
			lm_ptr_rgb += W;
		  }
	  }
	  else
	  {
		  ushort* dst_ptr = dst.ptr<ushort>();
		  ushort* dst_ptr_rgb = dst_rgb.ptr<ushort>();
		  for (int row = 0; row < 8; ++row)
		  {
			for (int col = 0; col < 8; ++col)
			{
			  dst_ptr[col] += lm_ptr[col];
			  dst_ptr_rgb[col] += lm_ptr_rgb[col];
			}
			dst_ptr += 8;
			lm_ptr += W;
			dst_ptr_rgb += 8;
			lm_ptr_rgb += W;
		  }
	  }
      
    }
  }
  //if(DEBUGGING) std::cout<<"dst16x16:"<<std::endl<<dst<<std::endl;
    //  if(DEBUGGING) std::cout<<"feature in local similarity: "<<countFeatureLocal<<std::endl;
}

void addUnaligned8u16u(const uchar * src1, const uchar * src2, ushort * res, int length)
{
  const uchar * end = src1 + length;

  while (src1 != end)
  {
    *res = *src1 + *src2;

    ++src1;
    ++src2;
    ++res;
  }
}

/**
 * \brief Accumulate one or more 8-bit similarity images.
 *
 * \param[in]  similarities Source 8-bit similarity images.
 * \param[out] dst          Destination 16-bit similarity image.
 */
void addSimilarities(const std::vector<Mat>& similarities, Mat& dst)
{
		
  if (similarities.size() == 1)
  {
	  //if(DEBUGGING) std::cout<<"8u = "<<CV_8U<<" - 16u = "<<CV_16U<<std::endl;
      //if(DEBUGGING) std::cout<<"sim0 = "<<similarities[0].type()<<std::endl;
    similarities[0].convertTo(dst, CV_16U);
  }
  else
  {
    // NOTE: add() seems to be rather slow in the 8U + 8U -> 16U case
    dst.create(similarities[0].size(), CV_16U);
    addUnaligned8u16u(similarities[0].ptr(), similarities[1].ptr(), dst.ptr<ushort>(), static_cast<int>(dst.total()));

    /// @todo Optimize 16u + 8u -> 16u when more than 2 modalities
    for (size_t i = 2; i < similarities.size(); ++i)
      add(dst, similarities[i], dst, noArray(), CV_16U);
  }
}

/****************************************************************************************\
*                               High-level Detector API                                  *
\****************************************************************************************/

Detector::Detector()
{
}

Detector::Detector(const std::vector< Ptr<Modality> >& modalities,
                   const std::vector<int>& T_pyramid)
  : modalities(modalities),
    pyramid_levels(static_cast<int>(T_pyramid.size())),
    T_at_level(T_pyramid)
{
}

void Detector::match(const std::vector<Mat>& sources, float threshold, std::vector<Match>& matches,
                     const std::vector<std::string>& class_ids, OutputArrayOfArrays quantized_images,
                     const std::vector<Mat>& masks) const
{
  matches.clear();
  if (quantized_images.needed())
    quantized_images.create(1, static_cast<int>(pyramid_levels * modalities.size()), CV_8U);

  assert(sources.size() == modalities.size());
  // Initialize each modality with our sources
  std::vector< Ptr<QuantizedPyramid> > quantizers;
  for (int i = 0; i < (int)modalities.size(); ++i){
    Mat mask, source;
    source = sources[i];
    if(!masks.empty()){
      assert(masks.size() == modalities.size());
      mask = masks[i];
    }
    assert(mask.empty() || mask.size() == source.size());
    quantizers.push_back(modalities[i]->process(source, mask));
  }
  // pyramid level -> modality -> quantization
  LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                 std::vector<LinearMemories>(modalities.size(), LinearMemories(8)));
  LinearMemoryPyramid lm_pyramid_rgb(pyramid_levels,
                                 std::vector<LinearMemories>(modalities.size(), LinearMemories(8)));

  // For each pyramid level, precompute linear memories for each modality
  std::vector<Size> sizes;
  for (int l = 0; l < pyramid_levels; ++l)
  {
    int T = T_at_level[l];
    std::vector<LinearMemories>& lm_level = lm_pyramid[l];
    std::vector<LinearMemories>& lm_level_rgb = lm_pyramid_rgb[l];


	//if(DEBUGGING) std::cout<<"quantizers size: "<<quantizers.size()<<std::endl;

    if (l > 0)
    {
      for (int i = 0; i < (int)quantizers.size(); ++i)
        quantizers[i]->pyrDown();
    }

    Mat quantized, spread_quantized, quantizedRGB, spread_quantized_rgb;
    std::vector<Mat> response_maps;
    std::vector<Mat> response_maps_rgb;
    for (int i = 0; i < (int)quantizers.size(); ++i)
    {
      quantizers[i]->quantize(quantized);
      quantizers[i]->quantizeRGB(quantizedRGB);
      spread(quantized, spread_quantized, T);
      spread(quantizedRGB, spread_quantized_rgb, T);
    //if(DEBUGGING) std::cout<<"immagine spread-quantized: "<<spread_quantized<<std::endl;
      computeResponseMaps(spread_quantized, response_maps);
      computeResponseMapsRGB(spread_quantized_rgb, response_maps_rgb);



 
      LinearMemories& memories = lm_level[i];
      LinearMemories& memories_rgb = lm_level_rgb[i];
      
      for (int j = 0; j < 8; ++j)
      {
        linearize(response_maps[j], memories[j], T);
        linearize(response_maps_rgb[j], memories_rgb[j], T);
      }

	
	  
      if (quantized_images.needed()) //use copyTo here to side step reference semantics.
        quantized.copyTo(quantized_images.getMatRef(static_cast<int>(l*quantizers.size() + i)));
    }
    

    
    sizes.push_back(quantized.size());
  }

  if (class_ids.empty())
  {
    // Match all templates
    TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
    for ( ; it != itend; ++it)
      matchClass(lm_pyramid, lm_pyramid_rgb, sizes, threshold, matches, it->first, it->second);
  }
  else
  {
    // Match only templates for the requested class IDs
    for (int i = 0; i < (int)class_ids.size(); ++i)
    {
      TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
      if (it != class_templates.end())
        matchClass(lm_pyramid, lm_pyramid_rgb, sizes, threshold, matches, it->first, it->second);
    }
  }

  // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
  std::sort(matches.begin(), matches.end());
  
  if(DEBUGGING) std::cout<<"matches size: "<<matches.size()<<std::endl;
  QuantizedPyramid * tmpQp = quantizers[0];
  ColorGradientPyramid * cgp = dynamic_cast<ColorGradientPyramid *>(tmpQp);
  
  
  if(DEBUGGING)imshow("magnitude in match", (1/cgp->magnitude)*255);
  if(DEBUGGING)imshow("magnitudeStrong in match", cgp->magnitudeStrong);
  

  
  
  
  //waitKey();
  std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
  matches.erase(new_end, matches.end());
  
  
  
  ///////////////DISEGNO SU MAGNITUDE/////////////////
if(matches.size() > 0 && signatureEnabled == true)
{
  static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < modalities.size(); ++m)
  {
    
    for(int scar = 0; scar<1; scar++)
    {
	cv::Scalar color = COLORS[m];
	const std::vector<cv::my_linemod::Template>& templates = getTemplates(matches[scar].class_id, matches[scar].template_id);
	//cv::Point offset = cv::Point(matches[matchesChecked].x, matches[matchesChecked].y);
	//cv::Point offsetEnd = cv::Point(matches[matchesChecked].x+templates[0].width, matches[matchesChecked].y+templates[0].height);
	cv::Point offset = cv::Point(matches[scar].x, matches[scar].y);
	
	
	
	//imshow("magnitudeStrong+contour",cgp->magnitudeStrong);
	
	
	cv::Point offsetEnd = cv::Point(matches[scar].x+templates[0].width, matches[scar].y+templates[0].height);
	//int T = getT(0);
	//Mat magnitudeMatched = cgp->magnitudeStrong(Rect(matches[matchesChecked].x, matches[matchesChecked].y, templates[0].width, templates[0].height));
	Mat magnitudeMatched = cgp->magnitudeStrong(Rect(matches[scar].x, matches[scar].y, templates[0].width, templates[0].height));
	
	
	Scalar colorM = Scalar( 255, 0, 0 );
	vector<vector<Point> > contours;
	contours.push_back(templates[0].contour);
	drawContours( magnitudeMatched, contours, 0, colorM);
	
	
	//imshow("cropped MASK", templates[0].croppedMask);
	if(DEBUGGING) imshow("magnitude Matched + contour", magnitudeMatched);

	int countMagDef = 0;
	for (int r = 0; r < magnitudeMatched.rows; ++r)
	{
	    float* source_r = magnitudeMatched.ptr<float>(r);
	    for (int c = 0; c < magnitudeMatched.cols; ++c)
	    {
		if(source_r[c] > 0)
		{
		    double res = pointPolygonTest( contours[0], Point(c,r), true);
		    if(res >= 0)
			countMagDef++;
		}
	    }
	}



	if(DEBUGGING) std::cout<<"countMagDef: "<<countMagDef<<std::endl;


	int diffFeatures = countMagDef - templates[0].totalFeatures;
	int allowed_more_mag = 1;
	int thresh_allow = (templates[0].totalFeatures/100)*allowed_more_mag;
	if(DEBUGGING) std::cout<<"total: "<<templates[0].totalFeatures<<" - magDef: "<<countMagDef<<" - diff: "<<diffFeatures<<std::endl;
	int impact = 0;
	if(diffFeatures > 0)
	    impact = diffFeatures/thresh_allow;
	
	if(DEBUGGING) std::cout<<" similarity prima: = "<<matches[scar].similarity<<std::endl;
	float similarityAdjusted = matches[scar].similarity - (float)impact;
	if(DEBUGGING) std::cout<<" similarity dopo: = "<<similarityAdjusted<<std::endl;
	if(DEBUGGING) std::cout<<" IMPACT = "<<impact<<std::endl;
	if(similarityAdjusted<threshold)
	{
	    //matches[matchesChecked].scartato = 1;
	    matches[scar].scartato = 1;
	}

	if(DEBUGGING) std::cout<<" scartato: "<<matches[scar].scartato<<std::endl;
    }
	

    

  }
}
//imshow("msource in match", magnitudeMatched);

////////////////////FINE DISEGNO SUMAGNITUDE/////////////////////
}

// Used to filter out weak matches
struct MatchPredicate
{
  MatchPredicate(float threshold) : threshold(threshold) {}
  bool operator() (const Match& m) { return m.similarity < threshold; }
  float threshold;
};

// Used to filter out weak RGB matches 
struct MatchPredicateRGB
{
  MatchPredicateRGB(float threshold) : threshold(threshold) {}
  bool operator() (const Match& m) { return m.similarity_rgb < threshold; }
  float threshold;
};





void Detector::matchClass(const LinearMemoryPyramid& lm_pyramid,
						  const LinearMemoryPyramid& lm_pyramid_rgb,
                          const std::vector<Size>& sizes,
                          float threshold, std::vector<Match>& matches,
                          const std::string& class_id,
                          const std::vector<TemplatePyramid>& template_pyramids) const
{
  // For each template...
  for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
  {
    const TemplatePyramid& tp = template_pyramids[template_id];

    // First match over the whole image at the lowest pyramid level
    /// @todo Factor this out into separate function
    const std::vector<LinearMemories>& lowest_lm = lm_pyramid.back();
    const std::vector<LinearMemories>& lowest_lm_rgb = lm_pyramid_rgb.back();

	//if(DEBUGGING) std::cout<<"lm_pyramid_Rgb size: "<<lm_pyramid_rgb.size()<<std::endl;
//if(DEBUGGING) std::cout<<std::endl<<"template_id: "<<template_id<<" - pyramid: "<<(Mat)lowest_lm[0]<<std::endl<<std::endl;

    // Compute similarity maps for each modality at lowest pyramid level
    std::vector<Mat> similarities(modalities.size());
    std::vector<Mat> similarities_rgb(1);
    int lowest_start = static_cast<int>(tp.size() - modalities.size());
    int lowest_T = T_at_level.back();
    int num_features = 0;
    int num_features_signature = 0;
    for (int i = 0; i < (int)modalities.size(); ++i)
    {
	  
      const Template& templ = tp[lowest_start + i];
      num_features += static_cast<int>(templ.features.size());
      num_features_signature += static_cast<int>(templ.featuresSignature.size());
      if(i == 0) //color modality, add rgb
	similarityRGB(lowest_lm[i], lowest_lm_rgb[i], templ, similarities[i], similarities_rgb[i], sizes.back(), lowest_T);
      if(i == 1) //depth modality, without rgb
	similarity(lowest_lm[i], templ, similarities[i], sizes.back(), lowest_T);
      //if(DEBUGGING) std::cout<<"matrice similarities "<<i<<":"<<std::endl<<similarities[0]<<std::endl;
    }


//if(DEBUGGING) std::cout<<std::endl<<"template_id: "<<template_id<<" - prima matrice: "<<similarities[0]<<std::endl<<std::endl;
    // Combine into overall similarity
    /// @todo Support weighting the modalities
    Mat total_similarity;
    addSimilarities(similarities, total_similarity);
    Mat total_similarity_rgb;
    addSimilarities(similarities_rgb, total_similarity_rgb);


    // Convert user-friendly percentage to raw similarity threshold. The percentage
    // threshold scales from half the max response (what you would expect from applying
    // the template to a completely random image) to the max response.
    // NOTE: This assumes max per-feature response is 4, so we scale between [2*nf, 4*nf].
    int raw_threshold = static_cast<int>(2*num_features + (threshold / 100.f) * (2*num_features) + 0.5f);
    int raw_threshold_rgb;
    if(punteggio16 == true)
	raw_threshold_rgb = static_cast<int>(8*num_features + (threshold / 100.f) * (8*num_features) + 0.5f);
    else
	raw_threshold_rgb = static_cast<int>(2*num_features + (threshold / 100.f) * (2*num_features) + 0.5f);
	//if(DEBUGGING) std::cout<<"("<<2<<"*"<<num_features<<")+("<<threshold<<"/"<<100<<")*("<<2<<"*"<<num_features<<")+"<<0.5<<std::endl;
	//if(DEBUGGING) std::cout<<"raw_threshold: "<<raw_threshold<<std::endl;

	//*DEBUG*//
	if(total_similarity.rows != total_similarity_rgb.rows || total_similarity.cols != total_similarity_rgb.cols)
		if(DEBUGGING) std::cout<<"ATTENZIONE LE MATRICI SIMILARITY NON SONO UGUALI"<<std::endl;

    //if(DEBUGGING) std::cout<<std::endl<<"template_id: "<<template_id<<"matrice: "<<total_similarity<<std::endl<<std::endl;

    // Find initial matches
    std::vector<Match> candidates;
    for (int r = 0; r < total_similarity.rows; ++r)
    {
      ushort* row = total_similarity.ptr<ushort>(r);
      ushort* row_rgb = total_similarity_rgb.ptr<ushort>(r);
      //std::cout<<"ushrot:"<<*row_rgb<<std::endl;
      for (int c = 0; c < total_similarity.cols; ++c)
      {
        int raw_score = row[c];
        int raw_score_rgb = row_rgb[c];//*modalities.size();
        
	//if(DEBUGGING) std::cout<<"template_id: "<<template_id<<"raw_score: "<<raw_score<<"raw_score_rgb: "<<raw_score_rgb<<std::endl;
	
        //raw_score = raw_score +raw_score_rgb;
        if (raw_score > raw_threshold)
        {
          int offset = lowest_T / 2 + (lowest_T % 2 - 1);
          int x = c * lowest_T + offset;
          int y = r * lowest_T + offset;
          float score = (raw_score * 100.f) / (4 * num_features) + 0.5f;
	  float score_rgb;
	  
	  if(punteggio16 == true)
	  {
	    if(featuresSignatureCandidates == false)
		score_rgb = (raw_score_rgb * 100.f) / (16 * num_features) + 0.5f;
	    else
		score_rgb = (raw_score_rgb * 100.f) / (16 * (num_features+num_features_signature)) + 0.5f;
	  }
	  else
	  {
	    if(featuresSignatureCandidates == false)
		score_rgb = (raw_score_rgb * 100.f) / (4 * num_features) + 0.5f;
	    else
		score_rgb = (raw_score_rgb * 100.f) / (4 * (num_features+num_features_signature)) + 0.5f;
          }
	  //if(DEBUGGING) std::cout<<"score temporaneo: "<<score<<"score rgb temporaneo: "<<score_rgb<<std::endl;
          candidates.push_back(Match(x, y, score, score_rgb, class_id, static_cast<int>(template_id), 0));
        }
      }
    }
    
	bool localEnabled = true;
    if(localEnabled == true)
    {
    
    //if(DEBUGGING) std::cout<<"dimensione candidates: "<<candidates.size()<<std::endl;
    /*for(int h = 0; h<candidates.size(); h++)
		if(DEBUGGING) std::cout<<"score candidates["<<h<<"]: "<<candidates[h].similarity<<"template.id: "<<candidates[h].template_id<<std::endl;
*/
    // Locally refine each match by marching up the pyramid
    for (int l = pyramid_levels - 2; l >= 0; --l)
    {
	//if(DEBUGGING) std::cout<<"l:"<<l<<std::endl;
      const std::vector<LinearMemories>& lms = lm_pyramid[l];
      const std::vector<LinearMemories>& lms_rgb = lm_pyramid_rgb[l];
      
      int T = T_at_level[l];
      int start = static_cast<int>(l * modalities.size());
      Size size = sizes[l];
      int border = 8 * T;
      int offset = T / 2 + (T % 2 - 1);
      int max_x = size.width - tp[start].width - border;
      int max_y = size.height - tp[start].height - border;

	  //local redefinition
      std::vector<Mat> similarities(modalities.size());
      std::vector<Mat> similarities_rgb(1);
      Mat total_similarity;
      Mat total_similarity_rgb;
      
      
      for (int m = 0; m < (int)candidates.size(); ++m)
      {
        Match& match = candidates[m];
        int x = match.x * 2 + 1; /// @todo Support other pyramid distance
        int y = match.y * 2 + 1;

        // Require 8 (reduced) row/cols to the up/left
        x = std::max(x, border);
        y = std::max(y, border);

        // Require 8 (reduced) row/cols to the down/left, plus the template size
        x = std::min(x, max_x);
        y = std::min(y, max_y);

        // Compute local similarity maps for each modality
        int num_features = 0;
	int num_features_signature = 0;
        for (int i = 0; i < (int)modalities.size(); ++i)
        {
          const Template& templ = tp[start + i];
          num_features += static_cast<int>(templ.features.size());
	  num_features_signature += static_cast<int>(templ.featuresSignature.size());
          
       /*   const Mat& m1 = (lms[i])[0];
          const Mat& m2 = (lms_rgb[i])[0];
          if(DEBUGGING) std::cout<<"i: "<<i<<" - lms[i].size(): "<<m1.cols<<" - lms_rgb[i].size(): "<<m2.cols<<std::endl;*/

          if(i == 0) //color modality, add rgb
	    similarityLocalRGB(lms[i], lms_rgb[i], templ, similarities[i], similarities_rgb[i], size, T, Point(x, y));
	  if(i == 1) //color modality, without rgb
	    similarityLocal(lms[i], templ, similarities[i], size, T, Point(x, y));
          //if(DEBUGGING) std::cout<<"matrice similarities DOPO LOCAL"<<i<<":"<<std::endl<<similarities[0]<<std::endl;
        }
        
        addSimilarities(similarities, total_similarity);
        addSimilarities(similarities_rgb, total_similarity_rgb);

        // Find best local adjustment
        int best_score = 0;
        int best_score_rgb = 0;
        int best_r = -1, best_c = -1;
        for (int r = 0; r < total_similarity.rows; ++r)
        {
          ushort* row = total_similarity.ptr<ushort>(r);
          ushort* row_rgb = total_similarity_rgb.ptr<ushort>(r);
          for (int c = 0; c < total_similarity.cols; ++c)
          {
            int score = row[c];
            int score_rgb = row_rgb[c]*modalities.size();
            /*if(DEBUGGING) std::cout<<"score: "<<score<<std::endl;
            if(DEBUGGING) std::cout<<"score_rgb: "<<score_rgb<<std::endl;
            */
            //if(score_rgb > best_score_rgb)
				//best_score_rgb = score_rgb;
            //ANDRA' OTTIMIZZATO, COSÌ LO SCORE È DEL 200%
            
	    
	    score = (score * 100.f) / (4 * num_features);
	    if(punteggio16 == true)
	    {
		if(featuresSignatureCandidates == false)
		    score_rgb = (score_rgb * 100.f) / (16 * num_features) + 0.5f;
		else
		    score_rgb = (score_rgb * 100.f) / (16 * (num_features+num_features_signature)) + 0.5f;
	    }
	    else
	    {
		if(featuresSignatureCandidates == false)
		    score_rgb = (score_rgb * 100.f) / (4 * num_features) + 0.5f;
		else
		    score_rgb = (score_rgb * 100.f) / (4 * (num_features+num_features_signature)) + 0.5f;
	    }
            //disabilitare abilitare score rgb
	    int score_combined = (score + score_rgb)/2;
	    
	    //score = ((score/100.0f)*90.0f) +((score_rgb/100.0f)*10.0f);
	    
	    
	    //HO INVERTITO SCORE_RGB CON SCORE!!!!!!!!!!!!!!!!!!!
	    
	    //if (score > best_score)
	    if (score_combined > best_score)
            {
		//if((score_rgb >= threshold && score >= threshold -5) || (score >= threshold && score_rgb >= threshold -5))
		//{
		    best_score = score;
		    best_score_rgb = score_rgb;
		    best_r = r;
		    best_c = c;
		//}
            }
          }
        }
        // Update current match
        match.x = (x / T - 8 + best_c) * T + offset;
        match.y = (y / T - 8 + best_r) * T + offset;
        //match.similarity = (best_score * 100.f) / (4 * num_features);
	match.similarity = best_score;
        //match.similarity_rgb = (best_score_rgb * 100.f) / (16 * num_features);
	match.similarity_rgb = best_score_rgb;
      }

	
//if(DEBUGGING) std::cout<<"dimensione candidates PRIMA IL REFINEMENT: "<<candidates.size()<<std::endl;
      // Filter out any matches that drop below the similarity threshold
      std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                            MatchPredicate(threshold));
    
      candidates.erase(new_end, candidates.end());

      std::vector<Match>::iterator new_end_rgb = std::remove_if(candidates.begin(), candidates.end(),
                                                            MatchPredicateRGB(threshold));
      candidates.erase(new_end_rgb, candidates.end());
      
      
      

//for(int h = 0; h<candidates.size(); h++)
//	if(DEBUGGING) std::cout<<"score candidates["<<h<<"] DOPO IL REFINEMENT: "<<candidates[h].similarity<<"template.id: "<<candidates[h].template_id<<std::endl;                                                     


    }
    
}//LOCALENABLED
	
    //if(DEBUGGING) std::cout<<"dimensione candidates DOPO IL REFINEMENT: "<<candidates.size()<<std::endl;
    matches.insert(matches.end(), candidates.begin(), candidates.end());
  }
}

int Detector::changeTemplate(const std::vector<Mat>& sources, const std::string& class_id,
                          const Mat& object_mask, int indexToChange, Rect* bounding_box)
{
	//if(DEBUGGING) std::cout<<"MI VEDI????????????????"<<std::endl;
  int num_modalities = static_cast<int>(modalities.size());
  std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());

  TemplatePyramid tp;
  tp.resize(num_modalities * pyramid_levels);

  // For each modality...
  for (int i = 0; i < num_modalities; ++i)
  {
    // Extract a template at each pyramid level
    Ptr<QuantizedPyramid> qp = modalities[i]->process(sources[i], object_mask);
    for (int l = 0; l < pyramid_levels; ++l)
    {
      /// @todo Could do mask subsampling here instead of in pyrDown()
      if (l > 0)
        qp->pyrDown();

      bool success = qp->extractTemplate(tp[l*num_modalities + i]);
      if (!success)
        return -1;
    }
  }

  Rect bb = cropTemplates(tp, object_mask);
  if (bounding_box)
    *bounding_box = bb;

  /// @todo Can probably avoid a copy of tp here with swap
  template_pyramids.erase(template_pyramids.begin()+indexToChange);
  template_pyramids.insert(template_pyramids.begin()+indexToChange, tp);
  return indexToChange;
}

int Detector::addTemplate(const std::vector<Mat>& sources, const std::string& class_id,
                          const Mat& object_mask, Rect* bounding_box)
{
  int num_modalities = static_cast<int>(modalities.size());
  std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());

  TemplatePyramid tp;
  tp.resize(num_modalities * pyramid_levels);
  // For each modality...
  for (int i = 0; i < num_modalities; ++i)
  {
    // Extract a template at each pyramid level
    Ptr<QuantizedPyramid> qp = modalities[i]->process(sources[i], object_mask);
    if(DEBUGGING) std::cout<<"process fatto"<<std::endl;
    
    for (int l = 0; l < pyramid_levels; ++l)
    {
      /// @todo Could do mask subsampling here instead of in pyrDown()
      if (l > 0)
      {
        qp->pyrDown();
	if(DEBUGGING) std::cout<<"pyrdown fatto"<<std::endl;
      }
      bool success = qp->extractTemplate(tp[l*num_modalities + i]);
    
    
    int countLabels [8] = {0,0,0,0,0,0,0,0};
    int countContours = 0;
    if(DEBUGGING)
    {
        for (int j = 0; j < (int)tp[l*num_modalities + i].features.size(); ++j)
	{
	  cv::my_linemod::Feature f = tp[l*num_modalities + i].features[j];
	  countLabels[f.rgbLabel]++;
	  if(f.onBorder)
	    countContours++;
	}
	std::cout<<"features count:"<<std::endl;
	std::cout<<"TOTALI: "<<(int)tp[l*num_modalities + i].features.size()<<"- SUL CONTORNO: "<<countContours<<std::endl;
    }
 
  if(DEBUGGING)
  {
      Mat tmp2(480, 640, CV_8UC3, Scalar(200, 200, 200));
      for(int k = 0; k<(int)tp[l*num_modalities + i].features.size(); k++)
      {
	  cv::Scalar colorT;
	      cv::my_linemod::Feature f = tp[l*num_modalities + i].features[k];
	      switch(f.rgbLabel)
	      {
	      case 0: colorT = CV_RGB(255,0,0); break;
	      case 1: colorT = CV_RGB(0,255,0); break;
	      case 2: colorT = CV_RGB(0,0,255); break;
	      case 3: colorT = CV_RGB(255,255,0); break;
	      case 4: colorT = CV_RGB(255,0,255); break;
	      case 5: colorT = CV_RGB(0,255,255); break;
	      case 6: colorT = CV_RGB(255,255,255); break;
	      case 7: colorT = CV_RGB(0,0,0); break;
	      }
	      cv::circle(tmp2, Point(f.x,f.y), 2, colorT);
	      //if(DEBUGGING) std::cout<<"y: "<<f.y<<std::endl;
      }
  }
  //imshow("template added",tmp2);
  //waitKey(0);
	
      if(DEBUGGING)
      {
	  for (int j = 0; j < 8; ++j)
	  {
	    if(j == 0)
		 std::cout<<"R: "<<countLabels[j]<<std::endl;
	      if(j == 1)
		 std::cout<<"G: "<<countLabels[j]<<std::endl;
	    if(j == 2)
		 std::cout<<"B: "<<countLabels[j]<<std::endl;
	    if(j == 3)
		 std::cout<<"RG: "<<countLabels[j]<<std::endl;
	    if(j == 4)
		 std::cout<<"RB: "<<countLabels[j]<<std::endl;
	    if(j == 5)
		 std::cout<<"GB: "<<countLabels[j]<<std::endl;
	    if(j == 6)
		 std::cout<<"RGB_WHITE: "<<countLabels[j]<<std::endl;
	    if(j == 7)
		 std::cout<<"RGB_BLACK: "<<countLabels[j]<<std::endl;
	    
	  }
	  std::cout<<std::endl;
      }
      if (!success)
        return -1;
    }
  }
  
  if(DEBUGGING) std::cout<<"PRIMA template width: "<<tp[0].width<<" - template.height: "<<tp[0].height<<std::endl;

  Rect bb = cropTemplates(tp, object_mask);
  if (bounding_box)
    *bounding_box = bb;
  
  if(DEBUGGING) std::cout<<"DOPO template width: "<<tp[0].width<<" - template.height: "<<tp[0].height<<std::endl;

  /// @todo Can probably avoid a copy of tp here with swap
  template_pyramids.push_back(tp);
  return template_id;
}

int Detector::addSyntheticTemplate(const std::vector<Template>& templates, const std::string& class_id)
{
  std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());
  template_pyramids.push_back(templates);
  return template_id;
}

const std::vector<Template>& Detector::getTemplates(const std::string& class_id, int template_id) const
{
  TemplatesMap::const_iterator i = class_templates.find(class_id);
  CV_Assert(i != class_templates.end());
  CV_Assert(i->second.size() > size_t(template_id));
  return i->second[template_id];
}

int Detector::numTemplates() const
{
  int ret = 0;
  TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
  for ( ; i != iend; ++i)
    ret += static_cast<int>(i->second.size());
  return ret;
}

int Detector::numTemplates(const std::string& class_id) const
{
  TemplatesMap::const_iterator i = class_templates.find(class_id);
  if (i == class_templates.end())
    return 0;
  return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
  std::vector<std::string> ids;
  TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
  for ( ; i != iend; ++i)
  {
    ids.push_back(i->first);
  }

  return ids;
}

void Detector::read(const FileNode& fn)
{
  setSingleton();
    
  class_templates.clear();
  pyramid_levels = fn["pyramid_levels"];
  fn["T"] >> T_at_level;

  modalities.clear();
  FileNode modalities_fn = fn["modalities"];
  FileNodeIterator it = modalities_fn.begin(), it_end = modalities_fn.end();
  for ( ; it != it_end; ++it)
  {
    modalities.push_back(Modality::create(*it));
  }
}

void Detector::write(FileStorage& fs) const
{
  fs << "pyramid_levels" << pyramid_levels;
  fs << "T" << T_at_level;

  fs << "modalities" << "[";
  for (int i = 0; i < (int)modalities.size(); ++i)
  {
    fs << "{";
    modalities[i]->write(fs);
    fs << "}";
  }
  fs << "]"; // modalities
}

  std::string Detector::readClass(const FileNode& fn, const std::string &class_id_override)
  {
  // Verify compatible with Detector settings
  FileNode mod_fn = fn["modalities"];
  CV_Assert(mod_fn.size() == modalities.size());
  FileNodeIterator mod_it = mod_fn.begin(), mod_it_end = mod_fn.end();
  int i = 0;
  for ( ; mod_it != mod_it_end; ++mod_it, ++i)
    CV_Assert(modalities[i]->name() == (std::string)(*mod_it));
  CV_Assert((int)fn["pyramid_levels"] == pyramid_levels);

  // Detector should not already have this class
    std::string class_id;
    if (class_id_override.empty())
    {
      std::string class_id_tmp = fn["class_id"];
      CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
      class_id = class_id_tmp;
    }
    else
    {
      class_id = class_id_override;
    }

  TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
  std::vector<TemplatePyramid>& tps = v.second;
  int expected_id = 0;

  FileNode tps_fn = fn["template_pyramids"];
  tps.resize(tps_fn.size());
  FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
  for ( ; tps_it != tps_it_end; ++tps_it, ++expected_id)
  {
    int template_id = (*tps_it)["template_id"];
    CV_Assert(template_id == expected_id);
    FileNode templates_fn = (*tps_it)["templates"];
    tps[template_id].resize(templates_fn.size());

    FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
    int i = 0;
    for ( ; templ_it != templ_it_end; ++templ_it)
    {
      tps[template_id][i++].read(*templ_it);
    }
  }

  class_templates.insert(v);
  return class_id;
}

void Detector::writeClass(const std::string& class_id, FileStorage& fs) const
{
  TemplatesMap::const_iterator it = class_templates.find(class_id);
  CV_Assert(it != class_templates.end());
  const std::vector<TemplatePyramid>& tps = it->second;

  fs << "class_id" << it->first;
  fs << "modalities" << "[:";
  for (size_t i = 0; i < modalities.size(); ++i)
    fs << modalities[i]->name();
  fs << "]"; // modalities
  fs << "pyramid_levels" << pyramid_levels;
  fs << "template_pyramids" << "[";
  for (size_t i = 0; i < tps.size(); ++i)
  {
    const TemplatePyramid& tp = tps[i];
    fs << "{";
    fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
    fs << "templates" << "[";
    for (size_t j = 0; j < tp.size(); ++j)
    {
      fs << "{";
      tp[j].write(fs);
      fs << "}"; // current template
    }
    fs << "]"; // templates
    fs << "}"; // current pyramid
  }
  fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string>& class_ids,
                           const std::string& format)
{
  for (size_t i = 0; i < class_ids.size(); ++i)
  {
    const std::string& class_id = class_ids[i];
    std::string filename = cv::format(format.c_str(), class_id.c_str());
    FileStorage fs(filename, FileStorage::READ);
    readClass(fs.root());
  }
}

void Detector::writeClasses(const std::string& format) const
{
  TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
  for ( ; it != it_end; ++it)
  {
    const std::string& class_id = it->first;
    std::string filename = cv::format(format.c_str(), class_id.c_str());
    FileStorage fs(filename, FileStorage::WRITE);
    writeClass(class_id, fs);
  }
}

static const int T_DEFAULTS[] = {5, 8};

Ptr<Detector> getDefaultLINE()
{ 
    setSingleton();
    
    std::vector< Ptr<Modality> > modalities;
    modalities.push_back(new ColorGradient);
    return new Detector(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}

Ptr<Detector> getDefaultLINEMOD()
{
  std::vector< Ptr<Modality> > modalities;
  modalities.push_back(new ColorGradient);
  modalities.push_back(new DepthNormal);
  return new Detector(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}

} // namespace linemod
} // namespace cv
