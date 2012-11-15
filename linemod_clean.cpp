#include <iterator>
#include <set>
#include <cstdio>
#include <sstream>
#include <iostream>

#include "test.h"

using namespace std;
using namespace cv;

static bool DEBUGGING = false;
static bool TESTING = true;
static int numPipelines = 8;
static int curMatch = 0;
static bool checkMinMaxBB = false;

// Adapted from cv_timer in cv_utilities
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

void readTemplateMap(map<string, string>& templateMap, string templateMapName);

void writeTemplateMap(map<string, string>& templateMap, string templateMapName);

cv::Ptr<cv::my_linemod::Detector> readLinemod(const std::string& filename);

void writeLinemod(const cv::Ptr<cv::my_linemod::Detector>& detector, const std::string& filename);

void drawResponse(const std::vector<cv::my_linemod::Template>& templates, 
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T, short scartato, string class_id, VideoFrame vf);

void checkAndWrite_bb_groundTruth(string class_id, Rect& bb, string immagineTemplate);

Mat rotateImage(const Mat& source, double angle);


int main(int argc, char * argv[])
{
    bool load = false;
    bool printTest = false;
    bool only1template = false;
    bool trainall = false;
    bool testall = false;
    bool trainselected = false;
    bool testselected = false;
    bool fromselected = false;
    bool resetresults = false;
    bool analyze = false;
    bool puliscirisultati = false;
    bool converted = false;
    bool analyzePipe = false;
    
    int num_modalities = 1;
    
    string filename;
    string templateMapName;

    for (int h = 1; h <= (argc - 1); h++) 
    {
       if (strcmp("-load", argv[h]) == 0) 
       {
          load = true;
          printf("load Template\n");
       }
       if (strcmp("-print", argv[h]) == 0) 
       {
          printTest = true;
          printf("Test video:\n");
       }
       if (strcmp("-1", argv[h]) == 0) 
       {   
          only1template = true;
          printf("ONOLY 1 TEMPLATE\n");
       }
       if (strcmp("-trainall", argv[h]) == 0) 
       {   
          trainall = true;
          printf("TRAIN ALL TEMPLATES:\n");
       }
       if (strcmp("-testall", argv[h]) == 0) 
       {   
          testall = true;
          printf("TEST EVERYTHING:\n");
       }
       if (strcmp("-trainselected", argv[h]) == 0) 
       {   
          trainselected = true;
          printf("TRAIN SELECTED:\n");
       }
       if (strcmp("-testselected", argv[h]) == 0) 
       {   
          testselected = true;
          printf("TEST SELECTED:\n");
       }
       if (strcmp("-fromselected", argv[h]) == 0) 
       {   
          fromselected = true;
          printf("-fromSelected\n");
       }
       if (strcmp("-resetresults", argv[h]) == 0) 
       {   
          resetresults = true;
          printf("WARNING --reset results ON--\n");
       }
       if (strcmp("-analyze", argv[h]) == 0) 
       {   
          analyze = true;
          printf("ANALYZE RESULTS\n");
       }
       if (strcmp("-analyzePipe", argv[h]) == 0) 
       {   
          analyzePipe = true;
          printf("ANALYZE PIPELINES\n");
       }
       if (strcmp("-pulisci", argv[h]) == 0) 
       {   
          puliscirisultati = true;
          printf("PULISCI RISULTATI\n");
       }
       if (strcmp("-converted", argv[h]) == 0) 
       {   
          converted = true;
          printf("CONVERTED\n");
       }
       if (strcmp("-debug", argv[h]) == 0) 
       {   
          DEBUGGING = true;
	  TESTING = false;
          printf("debugging\n");
       }
       
       
    }
    
    ////////////DA TOGLIERE/////////////77
    /*string filenameResult = "./debug/res.yml";
    cv::FileStorage fsResult(filenameResult, cv::FileStorage::APPEND);
    fsResult << "run" << "avviato";
    fsResult.release();*/
    ///////////////////////////////////////77
    
    string cartellaDiInput = "/media/TeraDati/HD Manu Clarabella/dataset rgbd/";
    string cartellaTemplate = "rgbd-dataset/";
    
    vector<string> cartelleVideo;
    vector<string> filesGroundTruth;
    vector<string> nomiVideo;
    vector<pair<string,string> > kitchen_small_1_categories;
    vector<pair<string,string> > desk_1_categories;
    vector<pair<string,string> > desk_2_categories;
    vector<pair<string,string> > desk_3_categories;
    vector<pair<string,string> > meeting_small_1_categories;
    vector<pair<string,string> > table_1_categories;
    vector<pair<string,string> > table_small_1_categories;
    vector<pair<string,string> > table_small_2_categories;
    
    vector<pair<string,string> > all_categories;
    
    SetVideos videoParams;
    
    //parametri dei templates
    //
    bool actual_use63; //default true
    int actual_featuresUsed; //default 63
    
    int actual_signFeat; //default 30
    
    int actual_threshold_rgb; //default 45
    
    //parametri matching
    //
    bool actual_punteggio16; //default true
    
    bool actual_featuresSignatureCandidates; //default true
    
    bool actual_grayEnabled;  //default true
    
    bool actual_signatureEnabled; //default true
    
    int actual_matching_threshold; //default 80
    
    videoParams.setDefaultParameters(actual_use63, actual_featuresUsed, actual_signFeat, actual_punteggio16, actual_featuresSignatureCandidates, actual_grayEnabled, actual_signatureEnabled, actual_threshold_rgb, actual_matching_threshold);
    
    SetTest testParameters = SetTest();
    
    vector<bool> v_use63 = testParameters.v_use63;
    vector<int> v_featuresUsed = testParameters.v_featuresUsed;
    
    vector<int> v_threshold_rgb = testParameters.v_threshold_rgb;
    vector<int> v_signFeat = testParameters.v_signFeat;
    
    
    vector<bool> v_punteggio16 = testParameters.v_punteggio16;
    
    vector<bool> v_featuresSignatureCandidates = testParameters.v_featuresSignatureCandidates;
    
    vector<bool> v_signatureEnabled = testParameters.v_signatureEnabled;
    
    
    vector<bool> v_grayEnabled = testParameters.v_grayEnabled;
    
    
    
    vector<int> v_matching_threshold = testParameters.v_matching_threshold;
    
    ///////////////
    /////-TRAIN-/////
    
    Timer total_train_timer;
    total_train_timer.start();
    
    if(trainall == true || trainselected == true)
    {
		
	if(trainselected == false)
	{
	    videoParams.setAll(converted, cartelleVideo, filesGroundTruth, nomiVideo, kitchen_small_1_categories, desk_1_categories, desk_2_categories, desk_3_categories, meeting_small_1_categories, table_1_categories, table_small_1_categories, table_small_2_categories, all_categories);
	}
	else//trainselected == true
	{

	    videoParams.setSelected(converted, cartelleVideo, filesGroundTruth, nomiVideo, all_categories);
	    
	    
	}
	for(int train1 = 0; train1<v_threshold_rgb.size(); train1++)
	{
	
	    if(trainselected == false)
		actual_threshold_rgb = v_threshold_rgb.at(train1);
	    
	for(int train2 = 0; train2<v_use63.size(); train2++)
	{
	    if(trainselected == false)
		actual_use63 = v_use63.at(train2);
		
	for(int train3 = 0; train3<v_featuresUsed.size(); train3++)
	{
	    //cambio il numero di feature solo se use63 è false
	    if(trainselected== false)
	    {
		if(actual_use63 == false)
		    actual_featuresUsed = v_featuresUsed.at(train3);
		else //use63 è true
		    actual_featuresUsed = 63;
	    }
	for(int train4 = 0; train4<v_signFeat.size(); train4++)
	{
	    if(trainselected == false)
		actual_signFeat = v_signFeat.at(train4);
		
	    cv::my_linemod::singleton_test * ptr_singleton = cv::my_linemod::singleton_test::get_instance();
	    ptr_singleton->initialized = true;
	    
	    ptr_singleton->s_threshold_rgb = actual_threshold_rgb;
	    ptr_singleton->s_use63 = actual_use63;
	    ptr_singleton->s_featuresUsed = actual_featuresUsed;
	    ptr_singleton->s_signFeat = actual_signFeat;
	    ptr_singleton->s_DEBUGGING = DEBUGGING;
	    ptr_singleton->s_numPipelines = numPipelines;
	    
	    for(int w = 0; w<all_categories.size(); w++)
	    {
		Timer train_timer;
		train_timer.start();
		cv::Ptr<cv::my_linemod::Detector> detector;
		detector = my_linemod::getDefaultLINE();
		
		vector<string> listFiles;
		pair<string,string> category = all_categories.at(w);
		map<string, string> templateMap;
		
		//ottengo tutti i nomi delle maschere nella cartella oggetto
		string cartellaOggetto = category.first + "/" + category.first + "_" + category.second + "/";
		string cartella = cartellaDiInput + cartellaTemplate + cartellaOggetto;
		readDirectory(cartella.c_str(), listFiles);
		getMasksFromListFile(listFiles);
		int size_files = listFiles.size();
		
		// Extract template
		string class_id = category.first + "_" + category.second;
		
		string string_use63;
		string string_threshold;
		string string_featuresUsed;
		string string_signFeat;
		string_use63 = "use63-" + boolToString(actual_use63);
		string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
		string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
		string_signFeat = "signFeat-" + intToString(actual_signFeat);
		
		if(trainselected == false)
		{
		    filename = "./templates/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
		    templateMapName = "./templates/" + class_id + "/" + "templatesMap_" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
		}
		else
		{
		    filename = "./selected/" + class_id + "_" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
		    templateMapName = "./selected/" + class_id + "_" + "templatesMap_" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
		}
		
		if(only1template == true)
		    size_files = 1;
		
		for(int i = 0; i<size_files; i++)
		{
		    
		    stringstream out;
		    out << i;
		    
		    string currentMask = listFiles.at(i);
		    
		    int index_temp = get_mask_number(currentMask);

		    if(index_temp%5==0)
		    {
			string immagineTemplate = cartellaDiInput + cartellaTemplate + cartellaOggetto + getColorFromMask(currentMask);
			string immagineMask = cartellaDiInput + cartellaTemplate + cartellaOggetto + currentMask;
			string immagineDepth = cartellaDiInput + cartellaTemplate + cartellaOggetto + getDepthFromMask(currentMask);
			
			if(DEBUGGING)
			    cout<<"img: "<<immagineTemplate<<std::endl;
			
			double resizes [6] = {0.7, 0.8, 0.9, 1, 1.2, 1.4};
			double rotations [3] = {-22.5, 1.0, 22.5};
			
			cv::Mat mask;
			mask = cv::imread(immagineMask, 0);
			if(mask.data != NULL)
			{
			    cv::Mat singleSource;
			    singleSource = cv::imread(immagineTemplate, 1);
			    cv::Mat singleSourceDepth;
			    singleSourceDepth = cv::imread(immagineDepth, 1);
			    
			    
			    for(int iter = 0; iter < 6; iter++)
			    {
				double resizeFactor = resizes[iter];
				cv::Mat singleSourceDst;
				cv::Mat maskDst;
				if(resizeFactor == 1.0)
				{
				    singleSourceDst = singleSource;
				    maskDst = mask;
				}
				else
				{
				    if(resizeFactor < 1.0)
				    {
					cv::resize(singleSource, singleSourceDst, Size(), resizeFactor, resizeFactor);//, CV_INTER_AREA);
					cv::resize(mask, maskDst, Size(), resizeFactor, resizeFactor);//, CV_INTER_AREA);
					
					/*cout<<"iter: "<< iter<<" - resizeFactor: "<<resizeFactor<<endl;
					imshow("original_"+intToString(iter), singleSource);
					imshow("rimpicciolito_"+intToString(iter), singleSourceDst);*/
					
				    }
				    else
				    {
					cv::resize(singleSource, singleSourceDst, Size(), resizeFactor, resizeFactor, CV_INTER_CUBIC);
					cv::resize(mask, maskDst, Size(), resizeFactor, resizeFactor, CV_INTER_CUBIC);
				    }
				}
				if(DEBUGGING)imshow("color_scaled_"+intToString(iter), singleSourceDst);
				for(int iterRot = 0; iterRot < 3; iterRot++)
				{
				    double rotationFactor = rotations[iterRot];
				    cv::Mat singleSourceFinal;
				    cv::Mat maskFinal;
				    if(rotationFactor == 1.0)
				    {
					singleSourceFinal = singleSourceDst.clone();
					maskFinal = maskDst;
				    }
				    else
				    {
					singleSourceFinal = rotateImage(singleSourceDst, rotationFactor);
					maskFinal = rotateImage(maskDst, rotationFactor);
				    }

				    cv::Rect bb;
				    std::vector<cv::Mat> sourcesTemplate;
				    sourcesTemplate.push_back(singleSourceFinal);
				    if(num_modalities == 2)
					sourcesTemplate.push_back(singleSourceDepth);
				    
				    
				    //memorizza scala nel class_id
				    //int template_id = detector->addTemplate(sourcesTemplate, class_id + "_scale_" + intToString(iter), maskDst, &bb);
				    int template_id = detector->addTemplate(sourcesTemplate, class_id, maskFinal, &bb);
				    
				    if(checkMinMaxBB == true)
					checkAndWrite_bb_groundTruth(class_id, bb, immagineTemplate);
				    
				    if(DEBUGGING == true && iterRot == 2)
				    {
					const std::vector<cv::my_linemod::Template>& templates = detector->getTemplates(class_id, template_id);
					for (int l = 0; l < (int)templates[0].features.size(); ++l)
					{
					  cv::Scalar colorT;
					  cv::my_linemod::Feature f = templates[0].features[l];
					  cv::Point pt(f.x + bb.x, f.y +bb.y);
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
					 //if(f.onBorder == true)
					    rectangle(singleSourceFinal, bb, CV_RGB(0,0,0), 2);
					    cv::circle(singleSourceFinal, pt, 2, colorT);
					}
					if(DEBUGGING)imshow("color_rotated_featured"+intToString(iterRot), singleSourceFinal);
					
					//imwrite("./provaTemplate_can_1.png", singleSourceDst);
					waitKey(0);
				    }
				    
				    printf("\n.....templating...");
				    cout<<class_id<<endl;
				    if (template_id != -1)
				    {
				      templateMap.insert(pair<string,string>(class_id + "_" + intToString(template_id),immagineTemplate));
				      
				      cout<<"*** Added template (id "<<template_id<<") for new object class "<<class_id<<" - path:"<<immagineTemplate<<"***"<<endl;
				      //printf("Extracted at (%d, %d) size %dx%d\n", bb.x, bb.y, bb.width, bb.height);
				    }

				}
				
			    }
			    //waitKey(0);
			    
			    
			}
			else
			{
			    if(DEBUGGING) cout<<immagineMask<<" non trovata"<<std::endl;
			}
		    }
		}
		    	
		writeLinemod(detector, filename);
		writeTemplateMap(templateMap, templateMapName);
		cout<<endl<<filename<<" salvato"<<endl;
		cout<<endl<<templateMapName<<" salvato"<<endl;
		train_timer.stop();
		cout<<"Train time: "<<train_timer.time()<<endl;
		if(trainselected == true)
		    return 0;
	    }
		    
	}		    
	}
	}
	}    
	total_train_timer.stop();
	cout<<"Whole train time: "<<total_train_timer.time()<<endl;
	
	return 0;
    }
    
    
	
    
    
    
    ///////////////
    /////-TEST-/////
    
    if(testselected == false) //testall == true
    {
	videoParams.setAll(converted, cartelleVideo, filesGroundTruth, nomiVideo, kitchen_small_1_categories, desk_1_categories, desk_2_categories, desk_3_categories, meeting_small_1_categories, table_1_categories, table_small_1_categories, table_small_2_categories, all_categories);
	
	//analyze funziona solo se testall è abilitato
	if(analyze == true)
	{
	    for(int nm = 0; nm < numPipelines; nm++)
	    {
		analyzeResults_category_video(nm, nomiVideo, v_threshold_rgb, v_use63, v_featuresUsed, v_signFeat, all_categories);
		analyzeResults_video(nm, nomiVideo, v_threshold_rgb, v_use63, v_featuresUsed, v_signFeat, all_categories);
		analyzeResults_global(nm, nomiVideo, v_threshold_rgb, v_use63, v_featuresUsed, v_signFeat, all_categories);
	    }
	    return 0;
	}
	if(analyzePipe == true)
	{
	    analyzeResults_pipelines(numPipelines);
	    return 0;
	}
	//resetResults funziona solo se testall è abilitato
	if(resetresults == true)
	    deleteResults(numPipelines, v_threshold_rgb, v_use63, v_featuresUsed, v_signFeat, nomiVideo, all_categories);
    }
    else//testselected == true
    {
	videoParams.setSelected(converted, cartelleVideo, filesGroundTruth, nomiVideo, all_categories);

    }
    
    Timer global_test_timer;
    global_test_timer.start();
    
    for(int test1 = 0; test1<v_threshold_rgb.size(); test1++)
    {
	if(testselected == false)
	    actual_threshold_rgb = v_threshold_rgb.at(test1);
	
    for(int test2 = 0; test2<v_use63.size(); test2++)
    {
	if(testselected == false)
	    actual_use63 = v_use63.at(test2);
	    
    for(int test3 = 0; test3<v_featuresUsed.size(); test3++)
    {
	if(testselected == false)
	{
	    //cambio il numero di feature solo se use63 è false
	    if(actual_use63 == false)
		actual_featuresUsed = v_featuresUsed.at(test3);
	    else
		actual_featuresUsed = 63;
	}
    for(int test4 = 0; test4<v_signFeat.size(); test4++)
    {
	if(testselected == false)
	    actual_signFeat = v_signFeat.at(test4);
    
    for(int test5 = 0; test5<v_punteggio16.size(); test5++)
    {
	if(testselected == false)
	    actual_punteggio16 = v_punteggio16.at(test5);
	    
    for(int test6 = 0; test6<v_featuresSignatureCandidates.size(); test6++)
    {
	if(testselected == false)
	    actual_featuresSignatureCandidates = v_featuresSignatureCandidates.at(test6);
    
    for(int test7 = 0; test7<v_signatureEnabled.size(); test7++)
    {
	if(testselected == false)
	    actual_signatureEnabled = v_signatureEnabled.at(test7);

    for(int test8 = 0; test8<v_grayEnabled.size(); test8++)
    {
	if(testselected == false)
	    actual_grayEnabled = v_grayEnabled.at(test8);
    
    for(int test9 = 0; test9<v_matching_threshold.size(); test9++)
    {
	if(testselected == false)
	    actual_matching_threshold = v_matching_threshold.at(test9);
	
    
	cv::my_linemod::singleton_test * ptr_singleton = cv::my_linemod::singleton_test::get_instance();
	ptr_singleton->initialized = true;
	
	ptr_singleton->s_threshold_rgb = actual_threshold_rgb;
	ptr_singleton->s_use63 = actual_use63;
	ptr_singleton->s_featuresUsed = actual_featuresUsed;
	ptr_singleton->s_signFeat = actual_signFeat;
	
	ptr_singleton->s_signatureEnabled = actual_signatureEnabled;
	ptr_singleton->s_grayEnabled = actual_grayEnabled;
	ptr_singleton->s_featuresSignatureCandidates = actual_featuresSignatureCandidates;
	ptr_singleton->s_matching_threshold = actual_matching_threshold;
	ptr_singleton->s_punteggio16 = actual_punteggio16;
	
	ptr_singleton->s_DEBUGGING = DEBUGGING;
	ptr_singleton->s_numPipelines = numPipelines;
	
	Timer one_video_loop_timer;
	one_video_loop_timer.start();
	
	for(int test10 = 0; test10<cartelleVideo.size(); test10++)
	{
	    string cartellaVideo;
	    string fileGroundTruth;
	    string nomeVideo;
	    
	    cartellaVideo = cartelleVideo.at(test10);
	    fileGroundTruth = filesGroundTruth.at(test10);
	    nomeVideo = nomiVideo.at(test10);

	    TestVideo tv = getTestVideo(fileGroundTruth);
	    if(printTest == true)
	    {
		printTestVideo(tv);
		return 0;
	    }
	    int framesNumber = tv.nFrames;
	    
	    string string_use63;
	    string string_threshold;
	    string string_featuresUsed;
	    string string_signFeat;
	    
	    string string_signatureEnabled;
	    string string_grayEnabled;
	    string string_featuresSignatureCandidates;
	    string string_matching;
	    string string_punteggio16;
	    
	    string_use63 = "use63-" + boolToString(actual_use63);
	    string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
	    string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
	    string_signFeat = "signFeat-" + intToString(actual_signFeat);

	    string_signatureEnabled = "signatureEnabled-" + boolToString(actual_signatureEnabled);
	    string_grayEnabled = "grayEnabled-" + boolToString(actual_grayEnabled);
	    string_featuresSignatureCandidates = "featuresSignCand-" + boolToString(actual_featuresSignatureCandidates);
	    string_matching = "matching-" + intToString(actual_matching_threshold);
	    string_punteggio16 = "punteggio16-" + boolToString(actual_punteggio16);
	    
	    vector<pair<string,string> > categoriesToCheck;
	    
	    if(testselected == true)
	    {
		//ci sarà solo una categoria
		categoriesToCheck.push_back(all_categories.front());
	    }
	    else
	    {
		string convertedString = "";
		if(converted == true)
		    convertedString = "converted/";
		    
		if(cartellaVideo == "video/rgbd-scenes/kitchen_small/kitchen_small_1/" + convertedString)
		    categoriesToCheck = kitchen_small_1_categories;
		else if(cartellaVideo == "video/rgbd-scenes/desk/desk_1/" + convertedString)
		    categoriesToCheck = desk_1_categories;
		else if(cartellaVideo == "video/rgbd-scenes/desk/desk_2/" + convertedString)
		    categoriesToCheck = desk_2_categories;
		else if(cartellaVideo == "video/rgbd-scenes/desk/desk_3/" + convertedString)
		    categoriesToCheck = desk_3_categories;
		else if(cartellaVideo == "video/rgbd-scenes/meeting_small/meeting_small_1/" + convertedString)
		    categoriesToCheck = meeting_small_1_categories;
		else if(cartellaVideo == "video/rgbd-scenes/table/table_1/" + convertedString)
		    categoriesToCheck = table_1_categories;
		else if(cartellaVideo == "video/rgbd-scenes/table_small/table_small_1/" + convertedString)
		    categoriesToCheck = table_small_1_categories;
		else if(cartellaVideo == "video/rgbd-scenes/table_small/table_small_2/" + convertedString)
		    categoriesToCheck = table_small_2_categories;
	    }
	    
	    for(int test11 = 0; test11<categoriesToCheck.size(); test11++)
	    {
		
		pair<string,string> category = categoriesToCheck.at(test11);
		string class_id = category.first + "_" + category.second;
	        
		
		if(puliscirisultati == true)
		    pulisci_risultati(numPipelines, actual_use63, actual_featuresUsed, actual_signFeat, actual_punteggio16, actual_featuresSignatureCandidates, actual_grayEnabled, actual_signatureEnabled, actual_threshold_rgb, actual_matching_threshold, class_id, nomeVideo);
		else
		{
		    vector<pair<string,string> > tmp_categoriesToCheck;
		    tmp_categoriesToCheck.push_back(category);
		    
		    vector<VideoResult> videoResults;
		    for(int nm= 0; nm<numPipelines; nm++)
			videoResults.push_back(VideoResult(cartellaVideo, tmp_categoriesToCheck));
		    
		    if(testselected == false || fromselected == false)
		    {
			filename = "./templates/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
			templateMapName = "./templates/" + class_id + "/" + "templatesMap_" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
		    }
		    else //fromselected == true, quindi lo prendo dalla cartella
		    {
			filename = "./selected/" + class_id + "_" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
			templateMapName = "./selected/" + class_id + "_" + "templatesMap_" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
		    }
		    
		    cv::Mat color, depth;
		    int num_classes = 1;
		    map<string, string> templateMap;
		    
		    cv::Ptr<cv::my_linemod::Detector> detector;
		    detector = readLinemod(filename);
		    
		    std::vector<std::string> ids = detector->classIds();
		    num_classes = detector->numClasses();
		    printf("\nLoaded %s with %d classes and %d templates\n",
			   argv[1], num_classes, detector->numTemplates());
		    if (!ids.empty())
		    {
		      printf("Class ids:\n");
		      std::copy(ids.begin(), ids.end(), std::ostream_iterator<std::string>(cout, "\n"));
		    }
		    
		    cout<<"Video:"<<nomeVideo<<endl;
		    readTemplateMap(templateMap, templateMapName);
	
	
		    int current = 1;
		    
		    /////////////////
		    /////-MATCH-/////
		    cout<<endl<<"FRAME ";
		    while (current <= framesNumber)
		    {
			cout<<"-"<<current<<flush; 
			stringstream out;
			out << current;

			string immagineRoi = cartellaDiInput + cartellaVideo + nomeVideo + "_" + out.str() + ".png";
			string immagineRoiDepth = cartellaDiInput + cartellaVideo + nomeVideo + "_" + out.str() + "_depth" + ".png";

			color = cv::imread(immagineRoi, 1);
			
			if(color.data != NULL)		    
			{
			    vector<cv::Mat> sources;
			    sources.push_back(color);
			    
			    if(num_modalities == 2)
			    {
				depth = cv::imread(immagineRoiDepth, 1);
				sources.push_back(depth);
			    }
				
			    cv::Mat display;
			    if(!TESTING)
				display = color.clone();   
			    
			    //drawGroundTruth(tv.frames.at(current-1), display);

			    // Perform matching
			    vector<vector<cv::my_linemod::Match> >matches;
			    vector<std::string> class_ids;
			    vector<cv::Mat> quantized_images;
			    
			    Timer match_timer;
			    match_timer.start();
			    detector->match(sources, actual_matching_threshold, matches, class_ids, quantized_images);
			    match_timer.stop();

			    int classes_visited = 0;
			    std::set<std::string> visited;
			    
			    for(int nm = 0; nm<numPipelines; nm++)
			    {
			    //elimina scartati
				vector<cv::my_linemod::Match>::iterator it;
				for (it=matches[nm].begin(); it<matches[nm].end(); it++)
				{
				    if((*it).scartato == 1)
				    {
				       matches[nm].erase(it);
				       break;
				    }
				    else
					break;
				}
			    
				//controllo falsi negativi e falsi positivi
				checkFrameFalses(matches[nm], tv.frames.at(current-1), tmp_categoriesToCheck, detector, videoResults[nm], display);
			    }
			    for (int i = 0; TESTING == false && (i < (int)matches[curMatch].size()) && (classes_visited < num_classes); ++i)
			    {
			      //VISUALIZZO SOLO IL CURMATCH DELLA PIPELINE
			      cv::my_linemod::Match m = matches[curMatch][i];

			      if (visited.insert(m.class_id).second)
			      {
				++classes_visited;
				  
				  printf("Similarity: %5.1f%%; Similarity_rgb: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d",
					 m.similarity, m.similarity_rgb, m.x, m.y, m.class_id.c_str(), m.template_id);
					 
				  if(templateMap.find(m.class_id + "_" + intToString(m.template_id)) != templateMap.end())
				  {

				    if(DEBUGGING) 
					{
					    Mat tempMatched;
					    tempMatched = imread((*templateMap.find(m.class_id + "_" + intToString(m.template_id))).second, 1);
					    imshow("template matchato:", tempMatched);
					    //cout<<", template name: "<< (*templateMap.find(m.class_id + "_" + intToString(m.template_id))).second<<endl;
					}
					else
					    if(DEBUGGING) cout<<endl;
				    }
				
				// Draw matching template
				if(!TESTING)
				{
				    const std::vector<cv::my_linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
				    drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), m.scartato, m.class_id.c_str(), tv.frames.at(current-1));
				}
				
			      }
			    }
			    
			    if (!TESTING && matches[curMatch].empty())
			      printf("No matches found...\n");
			    if (!TESTING)
			    {
			      printf("Matching: %.2fs\n", match_timer.time());
			    }
			    if (!TESTING)
			      printf("\n------------------------------------------------------------\n");
			    
			    if(!TESTING)
				cv::imshow("color_linemod_RGB", display);
			    
			    if(!TESTING && current == 1)
				waitKey(0);
			    
			    if(DEBUGGING)
				cv::imshow("normals", quantized_images[1]);

			    if(!TESTING)
			    {
				char key = (char)cvWaitKey(10);
				switch (key)
				{
				  case 'w':
				    // write model to disk
				    writeLinemod(detector, filename);
				    printf("Wrote detector and templates to %s\n", filename.c_str());
				    break;
				  case 'p':
				    // pause
				    waitKey();
				    break;
				  case 'q':
				    return 0;
				}
			    }
			}
			else
			{
			    if(DEBUGGING) cout<<immagineRoi<< "non trovata"<<std::endl;
			}
			current++;
		    }		
		    //SALVA RISULTATI
		    if(testselected == false)
			saveResults(actual_use63, actual_featuresUsed, actual_signFeat, actual_punteggio16, actual_featuresSignatureCandidates, actual_grayEnabled, actual_signatureEnabled, actual_threshold_rgb, actual_matching_threshold, videoResults, class_id, nomeVideo);
		    
		    if(testselected == true)
		    {
			for(int nm = 0; nm<numPipelines; nm++)
			{
			    cout<<"PIPELINE "<<nm<<endl;
			    printFalsesResult(videoResults[nm]);
			}
			one_video_loop_timer.stop();
			cout<<endl<<endl<<"1 VIDEO LOOP TIME: "<<one_video_loop_timer.time()<<endl<<endl;
			return 0;
		    }
		
		}
	    }
	}
	one_video_loop_timer.stop();
	cout<<endl<<endl<<"1 VIDEO LOOP TIME: "<<one_video_loop_timer.time()<<endl<<endl;
    }
    }
    }
    }
    }
    }
    }
    }
    }
    global_test_timer.stop();
    cout<<"Global test time: "<<global_test_timer.time()<<endl;
    
  return 0;
}



void drawResponse(const std::vector<cv::my_linemod::Template>& templates, 
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T, short scartato, string class_id, VideoFrame videoFrame)
{
  static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < num_modalities; ++m)
  {
    // NOTE: Original demo recalculated max response for each feature in the TxT
    // box around it and chose the display color based on that response. Here
    // the display color just depends on the modality.
    cv::Scalar color = COLORS[m];
    
    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      cv::my_linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      switch(f.rgbLabel)
      {
      case 0: color = CV_RGB(255,0,0); break;
      case 1: color = CV_RGB(0,255,0); break;
      case 2: color = CV_RGB(0,0,255); break;
      case 3: color = CV_RGB(255,255,0); break;
      case 4: color = CV_RGB(255,0,255); break;
      case 5: color = CV_RGB(0,255,255); break;
      case 6: color = CV_RGB(255,255,255); break;
      case 7: color = CV_RGB(0,0,0); break;
      } 
      
      cv::circle(dst, pt, T / 2, color);
    }
    //scartato = 0 - non scartato
    //scartato = 1 - feature interne maggiori di quelle aspettate
    //scartato = 2 - scartato con l'aggiunta di rgb
    //scartato = 3 - 
    
    /*
    cv::Point topLeft(offset.x, offset.y+templates[m].height);
    cv::Point bottomRight(offset.x+templates[m].width, offset.y);
    rectangle(dst, topLeft, bottomRight, COLORS[2], 2);
    */
    if(scartato == 1)
    {
        cv::line(dst, cv::Point(offset.x, offset.y+templates[m].height), cv::Point(offset.x+templates[m].width, offset.y), CV_RGB(255,0,0), 4);
        cv::line(dst, offset, cv::Point(offset.x+templates[m].width, offset.y+templates[m].height), CV_RGB(255,0,0), 4);
    }
    else
    {
        const cv::Scalar green = CV_RGB(0,255,0);
        const cv::Scalar red = CV_RGB(255,0,0);
        const cv::Scalar blue = CV_RGB(0,0,255);
        const cv::Scalar yellow = CV_RGB(255,255,0);
        
        for(int j = 0; j<videoFrame.nItem; j++)
        {
            Item item = videoFrame.items.at(j);
            string formattedCategory = getFormattedCategory(item.category, item.instance);
            if(DEBUGGING) cout<<endl<<"formatted: "<<formattedCategory<<endl;;
            if(DEBUGGING) cout<<"class_id: "<<class_id<<endl;
            if(formattedCategory == class_id)
            {
                Rect_<int> Bgt = Rect_<int>(item.left, item.top, item.right-item.left, item.bottom-item.top);
                Rect_<int> Ba = Rect_<int>(offset.x, offset.y, (offset.x+templates[m].width - offset.x), (offset.y+templates[m].height - offset.y));
                Rect_<int> intersection = Bgt & Ba;
                float areaIntersection = (float)intersection.area();
                float areaUnion = (float)(Bgt.area()+Ba.area())-areaIntersection;
                float result = areaIntersection/areaUnion;
                if(DEBUGGING) cout<<"intersezione: "<<areaIntersection<<endl;
                if(DEBUGGING) cout<<"union: "<<areaUnion<<endl;
                if(DEBUGGING) cout<<"result: "<<result<<endl;
                
                //disegno gt e intersezione buona
                rectangle(dst, cv::Point(Bgt.x,Bgt.y), cv::Point(Bgt.x+Bgt.width,Bgt.y+Bgt.height), blue, 2);
                rectangle(dst, cv::Point(Ba.x,Ba.y), cv::Point(Ba.x+Ba.width,Ba.y+Ba.height), yellow, 2);
                
                /*if(result >= 0.5)
                    cv::circle(dst, cv::Point(Ba.x+Ba.width/2,Ba.y+Ba.height/2), 15, green, -1);
                else
                    cv::circle(dst, cv::Point(Ba.x+Ba.width/2,Ba.y+Ba.height/2), 15, red, -1);
                */
            }
        }
    }
    
  }
}



void writeTemplateMap(map<string, string>& templateMap, string templateMapName)
{
    cv::FileStorage fs(templateMapName, cv::FileStorage::WRITE);
    map<string, string>::iterator it = templateMap.begin();
    CV_Assert(it != templateMap.end());

    fs << "templates" << "[";
    for(; it != templateMap.end(); ++it)
    {
        fs << "{";
        
        fs << "id" << (*it).first;
        fs << "path" << (*it).second;
        
        fs << "}"; // current template
    }
    fs << "]"; // templates
    //fs.releaseAndGetString();

}

void readTemplateMap(map<string, string>& templateMap, string templateMapName)
{
    cv::FileStorage fs(templateMapName, cv::FileStorage::READ);
    

    cv::FileNode fn = fs["templates"];
    for (cv::FileNodeIterator i = fn.begin(); i != fn.end(); ++i)
    {
        string id = (*i)["id"];
        string path = (*i)["path"];
        templateMap.insert(pair<string,string>(id,path));
    }
}

// Functions to store detector and templates in single XML/YAML file
cv::Ptr<cv::my_linemod::Detector> readLinemod(const std::string& filename)
{
  cv::Ptr<cv::my_linemod::Detector> detector = new cv::my_linemod::Detector;
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector->readClass(*i);

  return detector;
}

void writeLinemod(const cv::Ptr<cv::my_linemod::Detector>& detector, const std::string& filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs);

  std::vector<std::string> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int)ids.size(); ++i)
  {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}"; // current class
  }
  fs << "]"; // classes
  //fs.releaseAndGetString();
}

void checkAndWrite_bb_groundTruth(string class_id, Rect& bb, string immagineTemplate)
{
    int maxInt = std::numeric_limits<int>::max();
    
    string minWidthImg = "null";
    int minWidth = maxInt;
    
    string maxWidthImg = "null";
    int maxWidth = 0;
    
    string minHeightImg = "null";
    int minHeight = maxInt;
    
    string maxHeightImg = "null";
    int maxHeight = 0;
	
    string pathMeasures = "./templates/"+class_id+"/gtMeasures.yml";
    if(fileExists(pathMeasures.c_str()))
    {
	cv::FileStorage fs(pathMeasures, cv::FileStorage::READ);
    
	minWidth = (int)fs["minWidth"];
	minWidthImg = (string)fs["minWidthImg"];
	
	maxWidth = (int)fs["maxWidth"];
	maxWidthImg = (string)fs["maxWidthImg"];
	
	minHeight = (int)fs["minHeight"];
	minHeightImg = (string)fs["minHeightImg"];
	
	maxHeight = (int)fs["maxHeight"];
	maxHeightImg = (string)fs["maxHeightImg"];
	
	fs.release();
    }    
    
    
    bool changed = false;
    if(minWidth > bb.width)
    {
	minWidth = bb.width;
	minWidthImg = immagineTemplate;
	changed = true;
    }
    if(maxWidth < bb.width)
    {
	maxWidth = bb.width;
	maxWidthImg = immagineTemplate;
	changed = true;
    }
    if(minHeight > bb.height)
    {
	minHeight = bb.height;
	minHeightImg = immagineTemplate;
	changed = true;
    }
    if(maxHeight < bb.height)
    {
	maxHeight = bb.height;
	maxHeightImg = immagineTemplate;
	changed = true;
    }
    
    if(changed == true)
    {
	cv::FileStorage fs(pathMeasures, cv::FileStorage::WRITE);
	
	fs << "minWidth" << minWidth;
	fs << "minWidthImg" << minWidthImg;
	
	fs << "maxWidth" << maxWidth;
	fs << "maxWidthImg" << maxWidthImg;
	
	fs << "minHeight" << minHeight;
	fs << "minHeightImg" << minHeightImg;
	
	fs << "maxHeight" << maxHeight;
	fs << "maxHeightImg" << maxHeightImg;
	
	fs.release();
    }
    
    
}

Mat rotateImage(const Mat& source, double angle)
{
    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(source, dst, rot_mat, source.size(), INTER_CUBIC);
    return dst;
}
