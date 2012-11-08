#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h> // cvFindContours
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "objdetect_cv_modificato.h"
#include <iterator>
#include <set>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

class SetTest
{
public:
    SetTest(){
	
//////parametri che influenzano i template
//ogni vettore deve avere almeno un elemento

	v_use63.push_back(true);
	//v_use63.push_back(false);
	
	v_featuresUsed.push_back(126);
	
	//v_signFeat.push_back(20);
	//v_signFeat.push_back(25);
	v_signFeat.push_back(30);
	//v_signFeat.push_back(35);
	//v_signFeat.push_back(40);
	//v_signFeat.push_back(45);
	//v_signFeat.push_back(50);
	
	//v_threshold_rgb.push_back(15);
	//v_threshold_rgb.push_back(20);
	//v_threshold_rgb.push_back(25);
	//v_threshold_rgb.push_back(30);
	//v_threshold_rgb.push_back(35);
	//v_threshold_rgb.push_back(40);
	v_threshold_rgb.push_back(45); //trainato
	v_threshold_rgb.push_back(50); //trainato
	v_threshold_rgb.push_back(55); //trainato
	//v_threshold_rgb.push_back(60);
	//v_threshold_rgb.push_back(65);
	
//////parametri che influenzano solo il matching
//ogni vettore deve avere almeno un elemento	
	v_punteggio16.push_back(true);	
	v_punteggio16.push_back(false);
	
	v_featuresSignatureCandidates.push_back(true);
	v_featuresSignatureCandidates.push_back(false);
	
	v_signatureEnabled.push_back(true);
	//v_signatureEnabled.push_back(false);
	
	
	v_grayEnabled.push_back(true);
	v_grayEnabled.push_back(false);
	
	//v_matching_threshold.push_back(75);
	v_matching_threshold.push_back(80);
	//v_matching_threshold.push_back(85);
	
    }
    //parametri template
    //
    vector<bool> v_use63;
    vector<int> v_featuresUsed;
    
    vector<int> v_signFeat;
    //più bassa, meno i colori si accorpano
    vector<int> v_threshold_rgb;
    
    //parametri matching
    //
    vector<bool> v_punteggio16;
    
    vector<bool> v_featuresSignatureCandidates;
    
    vector<bool> v_signatureEnabled;
    
    
    vector<bool> v_grayEnabled; 
    
    

    vector<int> v_matching_threshold;

};

class SetVideos
{
public:
    SetVideos(){}
    void setDefaultParameters(bool& actual_use63, int& actual_featuresUsed, int& actual_signFeat, bool& actual_punteggio16, bool& actual_featuresSignatureCandidates, bool& actual_grayEnabled, bool& actual_signatureEnabled, int& actual_threshold_rgb, int& actual_matching_threshold)
    {
	//parametri dei template
	//
	actual_use63 = true; //default true
	actual_featuresUsed = 63; //default 63
	
	actual_signFeat = 30; //default 30
	
	actual_threshold_rgb = 50; //default 45
	
	//parametri del matching
	//	
	actual_punteggio16 = true; //default true
	
	actual_featuresSignatureCandidates = true; //default true
	
	actual_grayEnabled = true;  //default true
	
	actual_signatureEnabled = true; //default true
	
	actual_matching_threshold = 80; //default 80
    }
    void setSelected(bool isConverted, vector<string>& cartelleVideo, vector<string>& filesGroundTruth, vector<string>& nomiVideo, vector<pair<string,string> >& all_categories)
    {
	string converted = "";
	if(isConverted == true)
	    converted = "converted/";
	
	cartelleVideo.push_back("video/rgbd-scenes/kitchen_small/kitchen_small_1/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/kitchen_small/test.hdf5");
	nomiVideo.push_back("kitchen_small_1");
	
	//cartelleVideo.push_back("video/rgbd-scenes/desk/desk_1/" + converted);
	//filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/desk/test_desk_1.hdf5");
	//nomiVideo.push_back("desk_1");
	
	//cartelleVideo.push_back("video/rgbd-scenes/desk/desk_2/" + converted);
	//filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/desk/test_desk_2.hdf5");
	//nomiVideo.push_back("desk_2");
	
	//cartelleVideo.push_back("video/rgbd-scenes/desk/desk_3/" + converted);
	//filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/desk/test_desk_3.hdf5");
	//nomiVideo.push_back("desk_3");
	
	//cartelleVideo.push_back("video/rgbd-scenes/meeting_small/meeting_small_1/" + converted);
	//filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/meeting_small/test_meeting.hdf5");
	//nomiVideo.push_back("meeting_small_1");
	
	//cartelleVideo.push_back("video/rgbd-scenes/table/table_1/" + converted);
	//filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/table/test_table.hdf5");
	//nomiVideo.push_back("table_1");
	
	//cartelleVideo.push_back("video/rgbd-scenes/table_small/table_small_1/" + converted);
	//filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/table_small/test_table_small_1.hdf5");
	//nomiVideo.push_back("table_small_1");
	
	//cartelleVideo.push_back("video/rgbd-scenes/table_small/table_small_2/" + converted);
	//filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/table_small/test_table_small_2.hdf5");
	//nomiVideo.push_back("table_small_2");
	
	//all_categories.push_back(make_pair("bowl","2"));
	//all_categories.push_back(make_pair("bowl","3"));
	//all_categories.push_back(make_pair("bowl","4"));
	//all_categories.push_back(make_pair("cap","1"));
	//all_categories.push_back(make_pair("cap","3"));
	//all_categories.push_back(make_pair("cap","4"));
	//all_categories.push_back(make_pair("flashlight","1"));
	//all_categories.push_back(make_pair("flashlight","2"));
	//all_categories.push_back(make_pair("flashlight","3"));
	all_categories.push_back(make_pair("flashlight","5"));
	//all_categories.push_back(make_pair("soda_can","1"));
	//all_categories.push_back(make_pair("soda_can","3"));
	//all_categories.push_back(make_pair("soda_can","5"));
	//all_categories.push_back(make_pair("soda_can","6"));
	//all_categories.push_back(make_pair("coffee_mug","1"));
	//all_categories.push_back(make_pair("coffee_mug","5"));
	//all_categories.push_back(make_pair("coffee_mug","6"));
	//all_categories.push_back(make_pair("cereal_box","1"));
	//all_categories.push_back(make_pair("cereal_box","2"));
	//all_categories.push_back(make_pair("cereal_box","4"));
    }
    void setAll(bool isConverted, vector<string>& cartelleVideo, vector<string>& filesGroundTruth, vector<string>& nomiVideo, vector<pair<string,string> >& kitchen_small_1_categories, vector<pair<string,string> >& desk_1_categories, vector<pair<string,string> >& desk_2_categories, vector<pair<string,string> >& desk_3_categories, vector<pair<string,string> >& meeting_small_1_categories, vector<pair<string,string> >& table_1_categories, vector<pair<string,string> >& table_small_1_categories, vector<pair<string,string> >& table_small_2_categories, vector<pair<string,string> >& all_categories)
    {
	string converted = "";
	if(isConverted == true)
	    converted = "converted/";
	
	cartelleVideo.clear();
	filesGroundTruth.clear();
	nomiVideo.clear();
	kitchen_small_1_categories.clear();
	desk_1_categories.clear();
	desk_2_categories.clear();
	desk_3_categories.clear();
	meeting_small_1_categories.clear();
	table_1_categories.clear();
	table_small_1_categories.clear();
	table_small_2_categories.clear();
	
	
	cartelleVideo.push_back("video/rgbd-scenes/kitchen_small/kitchen_small_1/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/kitchen_small/test.hdf5");
	nomiVideo.push_back("kitchen_small_1");
	
	kitchen_small_1_categories.push_back(make_pair("flashlight","2"));
	kitchen_small_1_categories.push_back(make_pair("flashlight","5"));
	kitchen_small_1_categories.push_back(make_pair("soda_can","1"));
	kitchen_small_1_categories.push_back(make_pair("soda_can","6"));
	kitchen_small_1_categories.push_back(make_pair("coffee_mug","5"));
	kitchen_small_1_categories.push_back(make_pair("cereal_box","2"));
	kitchen_small_1_categories.push_back(make_pair("cap","1"));
	
	cartelleVideo.push_back("video/rgbd-scenes/desk/desk_1/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/desk/test_desk_1.hdf5");
	nomiVideo.push_back("desk_1");
	
	desk_1_categories.push_back(make_pair("soda_can","6"));
	desk_1_categories.push_back(make_pair("coffee_mug","5"));
	desk_1_categories.push_back(make_pair("cap","4"));
	
	cartelleVideo.push_back("video/rgbd-scenes/desk/desk_2/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/desk/test_desk_2.hdf5");
	nomiVideo.push_back("desk_2");
	
	desk_2_categories.push_back(make_pair("flashlight","1"));
        desk_2_categories.push_back(make_pair("soda_can","4"));
        //desk_2_categories.push_back(make_pair("bowl","3"));    
	
	cartelleVideo.push_back("video/rgbd-scenes/desk/desk_3/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/desk/test_desk_3.hdf5");
	nomiVideo.push_back("desk_3");
	
	//desk_3_categories.push_back(make_pair("bowl","4"));
	desk_3_categories.push_back(make_pair("flashlight","3"));
	desk_3_categories.push_back(make_pair("flashlight","5"));
	desk_3_categories.push_back(make_pair("cereal_box","1"));
	
	cartelleVideo.push_back("video/rgbd-scenes/meeting_small/meeting_small_1/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/meeting_small/test_meeting.hdf5");
	nomiVideo.push_back("meeting_small_1");
	
	//meeting_small_1_categories.push_back(make_pair("bowl","2"));
	//meeting_small_1_categories.push_back(make_pair("bowl","3"));
	meeting_small_1_categories.push_back(make_pair("cap","1"));
	meeting_small_1_categories.push_back(make_pair("cap","3"));
	meeting_small_1_categories.push_back(make_pair("flashlight","2"));
	meeting_small_1_categories.push_back(make_pair("flashlight","5"));
	meeting_small_1_categories.push_back(make_pair("soda_can","1"));
	meeting_small_1_categories.push_back(make_pair("soda_can","3"));
	meeting_small_1_categories.push_back(make_pair("soda_can","5"));
	meeting_small_1_categories.push_back(make_pair("coffee_mug","5"));
	meeting_small_1_categories.push_back(make_pair("coffee_mug","6"));
	meeting_small_1_categories.push_back(make_pair("cereal_box","1"));
	meeting_small_1_categories.push_back(make_pair("cereal_box","2"));
	    
	cartelleVideo.push_back("video/rgbd-scenes/table/table_1/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/table/test_table.hdf5");
	nomiVideo.push_back("table_1");
	
	//table_1_categories.push_back(make_pair("bowl","2"));
	table_1_categories.push_back(make_pair("cap","1"));
	table_1_categories.push_back(make_pair("cap","4"));
	table_1_categories.push_back(make_pair("cereal_box","4"));
	table_1_categories.push_back(make_pair("coffee_mug","1"));
	table_1_categories.push_back(make_pair("coffee_mug","4"));
	table_1_categories.push_back(make_pair("flashlight","3"));
	table_1_categories.push_back(make_pair("soda_can","4"));
	
	cartelleVideo.push_back("video/rgbd-scenes/table_small/table_small_1/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/table_small/test_table_small_1.hdf5");
	nomiVideo.push_back("table_small_1");
	
	//table_small_1_categories.push_back(make_pair("bowl","4"));
	table_small_1_categories.push_back(make_pair("cereal_box","1"));
	table_small_1_categories.push_back(make_pair("coffee_mug","1"));
	table_small_1_categories.push_back(make_pair("soda_can","3"));
	table_small_1_categories.push_back(make_pair("cap","1"));

	cartelleVideo.push_back("video/rgbd-scenes/table_small/table_small_2/" + converted);
	filesGroundTruth.push_back("/media/TeraDati/HD Manu Clarabella/dataset rgbd/video/rgbd-scenes/table_small/test_table_small_2.hdf5");
	nomiVideo.push_back("table_small_2");
	
	table_small_2_categories.push_back(make_pair("cap","4"));
	table_small_2_categories.push_back(make_pair("cereal_box","4"));
	table_small_2_categories.push_back(make_pair("soda_can","1"));
	
	//aggiungo tutte le singole categorie in un unico vettore
	for(int c1 = 0; c1<kitchen_small_1_categories.size(); c1++)
	    all_categories.push_back(kitchen_small_1_categories.at(c1));
	for(int c1 = 0; c1<desk_1_categories.size(); c1++)
	    all_categories.push_back(desk_1_categories.at(c1));
	for(int c1 = 0; c1<desk_2_categories.size(); c1++)
	    all_categories.push_back(desk_2_categories.at(c1));
	for(int c1 = 0; c1<desk_3_categories.size(); c1++)
	    all_categories.push_back(desk_3_categories.at(c1));
	for(int c1 = 0; c1<meeting_small_1_categories.size(); c1++)
	    all_categories.push_back(meeting_small_1_categories.at(c1));
	for(int c1 = 0; c1<table_1_categories.size(); c1++)
	    all_categories.push_back(table_1_categories.at(c1));
	for(int c1 = 0; c1<table_small_1_categories.size(); c1++)
	    all_categories.push_back(table_small_1_categories.at(c1));
	for(int c1 = 0; c1<table_small_2_categories.size(); c1++)
	    all_categories.push_back(table_small_2_categories.at(c1));
	//elimino i doppioni
	sort(all_categories.begin(), all_categories.end());
	all_categories.erase( unique( all_categories.begin(), all_categories.end() ), all_categories.end() );
	
	//PER ADDESTRARE SOLO LE CATEGORIE VOLUTE//
	//all_categories.clear();
	//all_categories.push_back(make_pair("flashlight","1"));
	//////////////////////////////////////////
	
	
    }
    

};


class Item
{
public:
    Item(string category, int instance, int bottom, int top, int left, int right) : category(category), instance(instance), bottom(bottom), top(top), left(left), right(right) 
    {
	stringstream ss;
	string b, t, l, r;
	ss<<bottom;
	ss>>b;
	ss<<top;
	ss>>t;
	ss<<left;
	ss>>l;
	ss<<right;
	ss>>r;
	id = b+t+l+r;
    }
    string category;
    int instance;
    int bottom;
    int top;
    int left;
    int right;
    string id;
};

namespace std
{
    template<> struct less<Item>
    {
       bool operator() (const Item& lhs, const Item& rhs)
       {
           return lhs.id < rhs.id;
       }
    };
}


class VideoFrame
{
public:
    VideoFrame(){}
    VideoFrame(int nI, int fN) : nItem(nI), frameNumber(fN) {}
    int nItem;
    int frameNumber;
    vector<Item> items;
};

class TestVideo
{
public:
    TestVideo(string vF, int nF) : videoFilename(vF), nFrames(nF) {}
    string videoFilename;
    int nFrames;
    vector<VideoFrame> frames;
    vector<string> categories;
};

class FalsePositive
{
public:
    FalsePositive(string classFound, float matchScore, int frameNumber, Rect_<int> boundingBox) : classFound(classFound), matchScore(matchScore), frameNumber(frameNumber), boundingBox(boundingBox) {}
    string classFound;
    float matchScore;
    int frameNumber;
    Rect_<int> boundingBox;
    
};

class FalseNegative
{
public:
    FalseNegative(string classExpected, int frameNumber, Rect_<int> boundingBox) : classExpected(classExpected), frameNumber(frameNumber), boundingBox(boundingBox) {}
    string classExpected;
    int frameNumber;
    Rect_<int> boundingBox;
};

class CategoryResult
{
public:
    CategoryResult(string name) : name(name) {
	nFalseNegative = 0;
	nFalsePositive = 0;
    }
    string name;
    int nFalseNegative;
    int nFalsePositive;
    vector<FalseNegative> arrayFalseNegative;
    vector<FalsePositive> arrayFalsePositive;
};

class VideoResult
{
public:
    VideoResult(string name, vector<pair<string,string> > categoriesToCheck) : name(name) 
    {
	nFalseNegative = 0;
	nFalsePositive = 0;
	positivesExpected = 0;
	
	for(int i = 0; i<categoriesToCheck.size(); i++)
	{
	    string formattedCategory = categoriesToCheck.at(i).first + "_" + categoriesToCheck.at(i).second;
	    CategoryResult cr = CategoryResult(formattedCategory);
	    mapCategoryResult.insert(make_pair(formattedCategory, cr));
	    categoriesChecked.push_back(formattedCategory);
	}
    }
    string name;
    int nFalseNegative;
    int nFalsePositive;
    int positivesExpected;
    map<string, CategoryResult> mapCategoryResult;
    vector<string> categoriesChecked;
};


void printTestVideo(TestVideo tv);
string last_occur(const char* str1, const char* str2);
string charToString(char* c);
string charToString_alphanumericFilter(char* c);
string intToString(int number);
int stringToInt(string s);
string boolToString(bool boolean);
const char * stringToChar(string str);
int get_mask_number(string mask);
string getFrameNumber(int number, int nElements);
string getItemNumber(int number, int nElements);
string getFormattedCategory(string category, int instance);
TestVideo getTestVideo(string pathVideoMat);
void drawGroundTruth(VideoFrame videoFrame, cv::Mat& dst);
void checkFrameFalses(vector<my_linemod::Match> matches, VideoFrame videoFrame, vector<pair<string,string> > categoriesToCheck, cv::Ptr<my_linemod::Detector> detector, VideoResult& videoResult, Mat& dst);
void printFalsesResult(VideoResult vr);
bool fileExists(const char * path);
void readDirectory(const char* dirname, vector<string>& listFiles);
void getMasksFromListFile(vector<string>& listFiles);
string getColorFromMask(string mask);
string getDepthFromMask(string mask);
void saveResults(bool& actual_use63, int& actual_featuresUsed, int& actual_signFeat, bool& actual_punteggio16, bool& actual_featuresSignatureCandidates, bool& actual_grayEnabled, bool& actual_signatureEnabled, int& actual_threshold_rgb, int& actual_matching_threshold, VideoResult videoResult, string class_id, string nomeVideo);
void deleteResults(vector<int> v_threshold_rgb,vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<string> nomiVideo, vector<pair<string,string> > all_categories);
void analyzeResults_category_video(vector<string>& nomiVideo, vector<int> v_threshold_rgb,vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<pair<string,string> > all_categories);
void analyzeResults_video(vector<string>& nomiVideo, vector<int> v_threshold_rgb, vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<pair<string,string> > all_categories);
void analyzeResults_global(vector<string>& nomiVideo, vector<int> v_threshold_rgb, vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<pair<string,string> > all_categories);
void pulisci_risultati(bool& actual_use63, int& actual_featuresUsed, int& actual_signFeat, bool& actual_punteggio16, bool& actual_featuresSignatureCandidates, bool& actual_grayEnabled, bool& actual_signatureEnabled, int& actual_threshold_rgb, int& actual_matching_threshold, string class_id, string nomeVideo);
