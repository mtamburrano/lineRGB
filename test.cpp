

#include "hdf5.h"
#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif


#include "H5Cpp.h"
#include "hdf5_hl.h"

#include "test.h"


using namespace std;
using namespace cv;

static bool discardGroundTruth = true;



size_t getInMemDataSize(hid_t id)
{
    const char *func = "Attribute::getInMemDataSize";

    // Get the data type of this attribute
    hid_t mem_type_id = H5Dget_type(id);
    if( mem_type_id < 0 )
    {
	throw AttributeIException(func, "H5Aget_type failed");
    }

    // Get the data type's size by first getting its native type then getting
    // the native type's size.
    hid_t native_type = H5Tget_native_type(mem_type_id, H5T_DIR_DEFAULT);
    if (native_type < 0)
    {
	throw AttributeIException(func, "H5Tget_native_type failed");
    }
    size_t type_size = H5Tget_size(native_type);
    if (type_size == 0)
    {
	throw AttributeIException(func, "H5Tget_size failed");
    }

    // Close the native type and the datatype of this attribute.
    if (H5Tclose(native_type) < 0)
    {
	throw DataSetIException(func, "H5Tclose(native_type) failed");
    }
    if (H5Tclose(mem_type_id) < 0)
    {
	throw DataSetIException(func, "H5Tclose(mem_type_id) failed");
    }

    // Get number of elements of the attribute by first getting its dataspace
    // then getting the number of elements in the dataspace
    hid_t space_id = H5Dget_space(id);
    if (space_id < 0)
    {
	throw AttributeIException(func, "H5Aget_space failed");
    }
    hssize_t num_elements = H5Sget_simple_extent_npoints(space_id);
    if (num_elements < 0)
    {
	throw AttributeIException(func, "H5Sget_simple_extent_npoints failed");
    }

    // Close the dataspace
    if (H5Sclose(space_id) < 0)
    {
	throw DataSetIException(func, "H5Sclose failed");
    }

    // Calculate and return the size of the data
    size_t data_size = type_size * num_elements;
    return(data_size);
}

herr_t file_info(hid_t loc_id, const char *name, void *opdata)
{
    H5G_stat_t statbuf;

    /*
     * Get type of the object and display its name and type.
     * The name of the object is passed to this function by 
     * the Library. Some magic :-)
     */
    H5O_info_t* infos = (H5O_info_t*) malloc(sizeof(H5O_info_t));
    H5Oget_info(loc_id, infos);
    
     
    H5Gget_objinfo(loc_id, name, false, &statbuf);
    switch (statbuf.type) {
    case H5G_GROUP: 
         printf(" Object with name %s is a group \n", name);
         break;
    case H5G_DATASET: 
         printf(" Object with name %s is a dataset \n", name);
         break;
    case H5G_TYPE: 
         printf(" Object with name %s is a named datatype \n", name);
         break;
    default:
         printf(" Unable to identify an object ");
    }
    return 0;
 }

void printFalseNegative(FalseNegative fn, int index)
{
    if(index >= 0)
        cout<<"       "<<index<<": "<<endl;
    cout<<"         Frame number: "<<fn.frameNumber<<endl;
}

void printFalsePositive(FalsePositive fp, int index)
{
    if(index >= 0)
        cout<<"       "<<index<<": "<<endl;
    cout<<"         Frame number: "<<fp.frameNumber<<endl;
    cout<<"         Match score: "<<fp.matchScore<<endl;
}

void printFalsesResult(VideoResult vr)
{
    cout<<"Name: "<<vr.name<<endl;
    cout<<"Positivi attesi: "<<vr.positivesExpected<<endl;
    cout<<"False Positives: "<<vr.nFalsePositive<<endl;
    cout<<"False Negatives: "<<vr.nFalseNegative<<endl;
    /*cout<<"Categorie contenute: "<<endl;
    
    
    for(int k = 0; k<vr.categoriesChecked.size(); k++)
    {
        string actualCategory = vr.categoriesChecked.at(k);
        cout<<endl<<"  Category: "<<actualCategory<<":"<<endl;
        map<string, CategoryResult> ::iterator itCat;
        itCat = vr.mapCategoryResult.find(actualCategory);
        CategoryResult cr = (*itCat).second;
        
        cout<<"    Falses Positives: "<<cr.nFalsePositive<<endl;
        for(int i = 0; i<cr.nFalsePositive; i++)
            printFalsePositive(cr.arrayFalsePositive.at(i), i);
            
        cout<<"    Falses Negatives: "<<cr.nFalseNegative<<endl;
        for(int i = 0; i<cr.nFalseNegative; i++)
            printFalseNegative(cr.arrayFalseNegative.at(i), i);
        //VideoFrame vf = tv.frames.at(i);
        //cout<<"  Oggetti contenuti: "<<vf.nItem<<endl;
        
    }*/
}

void printTestVideo(TestVideo tv)
{
    
    for(int i = 0; i<tv.nFrames; i++)
    {
        cout<<endl<<"  Frame "<<i<<":"<<endl;
        VideoFrame vf = tv.frames.at(i);
        cout<<"  Oggetti contenuti: "<<vf.nItem<<endl;
        
        for(int j = 0; j<vf.nItem; j++)
        {
            cout<<endl<<"    Oggetto "<<j<<":"<<endl;
            Item item = vf.items.at(j);
            cout<<"    Category: "<<item.category<<endl;
            cout<<"    Instance: "<<item.instance<<endl;
            cout<<"    Bottom: "<<item.bottom<<endl;
            cout<<"    Top: "<<item.top<<endl;
            cout<<"    Left: "<<item.left<<endl;
            cout<<"    Right: "<<item.right<<endl;
            
        }
        
    }
    
    cout<<"Path: "<<tv.videoFilename<<endl;
    cout<<"Numero di Frame: "<<tv.nFrames<<endl;
    cout<<"Categorie contenute: "<<endl;
    for(int k = 0; k<tv.categories.size(); k++)
    {
        cout<<"  "<<tv.categories.at(k)<<endl;
    }
    
}

string last_occur(const char* str1, const char* str2)
{
  char* strp;
  int len1, len2;

  len2 = strlen(str2);
  if(len2==0)
    return (char*)str1;

  len1 = strlen(str1);
  if(len1 - len2 <= 0)
    return 0;
   
  strp = (char*)(str1 + len1 - len2);
  while(strp != str1)
  {
    if(*strp == *str2)
    {
      if(strncmp(strp,str2,len2)==0)
        return charToString(strp);
    }
    strp--;
  }
  return 0;
}

int get_mask_number(string mask)
{
    string tmp = last_occur(mask.c_str(), "_");
    
    string cutted = mask.substr(0, mask.size()-tmp.size());
    string final = last_occur(cutted.c_str(), "_");
    final = final.substr(1, final.size());
    
    return stringToInt(final);
}

string charToString(char* c)
{
    stringstream ss;
    string s;
    ss << c;
    ss >> s;
    
    return s;
}

bool isAlphaNumeric(char c)
{
    if(c == '_')
        return false;
    else
        return !isalnum(c);
}

string boolToString(bool boolean)
{
    if(boolean == true)
        return "true";
    else
        return "false";
}

string charToString_alphanumericFilter(char* c)
{
    stringstream ss;
    string s;
    ss << c;
    ss >> s;
    
    s.erase(std::remove_if(s.begin(), s.end(), isAlphaNumeric), s.end());
    
    return s;
}

int stringToInt(string s)
{
    stringstream ss;
    int number;
    ss << s;
    ss >> number;
    
    return number;
}

string intToString(int number)
{
    stringstream ss;
    string s;
    ss << number;
    ss >> s;
    
    return s;
}

const char * stringToChar(string str)
{
    /*char *buf;
    buf = (char*)malloc(str.size());

    strcpy(buf, str.c_str());

    return buf;*/
    return str.c_str();
}

string getFrameNumber(int number, int nElements)
{
    string finalString;
    stringstream ss;
    ss << number;
    if(nElements < 100)
    {
        if(number < 10 )
            finalString = "_0" + ss.str();
        else
            finalString = "_" + ss.str();
    }
    else if(nElements > 99 && nElements < 1000)
    {
        if(number < 10)
            finalString = "_00" + ss.str();
        else if(number < 100 )
            finalString = "_0" + ss.str();
        else
            finalString = "_" + ss.str();
    }
    else if(nElements > 999)
    {
        if(number < 10)
            finalString = "_000" + ss.str();
        else if(number < 100 )
            finalString = "_00" + ss.str();
        else if(number < 1000 )
            finalString = "_0" + ss.str();
        else
            finalString = "_" + ss.str();
    }
    
    
    return finalString;
}

string getItemNumber(int number, int nElements)
{
    string finalString;
    stringstream ss;
    ss << number;
    
    if(nElements < 10)
    {
        finalString = "_" + ss.str();
    }
    else if(nElements < 100)
    {
        if(number < 10)
            finalString = "_0" + ss.str();
        else
            finalString = "_" + ss.str();
    }
        
    return finalString;
}

string getFormattedCategory(string category, int instance)
{
    stringstream ss;
    string instanceString;
    ss << instance;
    ss >> instanceString;
    return (category + "_" + instanceString);
}

TestVideo getTestVideo(string pathVideoMat)
{
    
    int frameSize;
    string frameFields[6] = {"category","instance","bottom","top","left","right"};
    
    hid_t fileIter = H5Fopen( stringToChar(pathVideoMat), H5F_ACC_RDONLY, H5P_DEFAULT );
    hid_t datasetBboxesDims = H5Dopen(fileIter, "bboxes/value/dims");
    
    int * bufInt;
    
    
    bufInt = (int*)malloc(sizeof(int));
    H5Dread(datasetBboxesDims, H5T_NATIVE_INT, H5S_ALL, 
                  H5S_ALL, H5P_DEFAULT, bufInt); 
    frameSize = *bufInt;
    H5Dclose(datasetBboxesDims);
    
    TestVideo tv = TestVideo(pathVideoMat, frameSize);
    vector<VideoFrame> vFrames = vector<VideoFrame>();
    vector<string> tvCategories = vector<string>();
    
    string framePrefix = "bboxes/value/";
    for(int i=0; i<frameSize; i++)
    {
        

        string actualFrameType = framePrefix + getFrameNumber(i, frameSize);

        hid_t groupactualFrameType = H5Gopen(fileIter, stringToChar(actualFrameType));
        H5O_info_t* infos = (H5O_info_t*) malloc(sizeof(H5O_info_t));
        H5Oget_info(groupactualFrameType, infos);
        H5Gclose(groupactualFrameType);   
        int numAttrs = (int)infos->num_attrs; 
        
        VideoFrame vf;
        
        if(numAttrs == 2)
        {
            vf = VideoFrame(0, i);
            vFrames.push_back(vf);
        }
        else //nel frame ci sono oggetti
        {
            int nItems;
            string actualFrame = framePrefix + getFrameNumber(i, frameSize) + "/value/";
            string actualFrameDims = actualFrame + "bottom/value/dims";
            hid_t datasetactualFrameDims = H5Dopen(fileIter, stringToChar(actualFrameDims));
            
            
            bufInt = (int*)malloc(sizeof(int));
            H5Dread(datasetactualFrameDims, H5T_NATIVE_INT, H5S_ALL, 
                      H5S_ALL, H5P_DEFAULT, bufInt); 
            nItems = *bufInt;
            H5Dclose(datasetactualFrameDims);
            
            vf = VideoFrame(nItems, i+1);
            vector<Item> vfItems = vector<Item>();
            vector<Item> vfDiscardedItems = vector<Item>();
            
            for (int j = 0; j<nItems; j++)
            {
                string Icategory;
                float Iinstance;
                float Ibottom;
                float Itop;
                float Ileft;
                float Iright;
                float* bufFloat;
                for (int k = 0; k < 6; k++)
                {
                    
                    
                    string attributePath = actualFrame + frameFields[k] +"/value/" + getItemNumber(j, nItems) + "/value";                   

                 
                    if(k == 0)
                    {   
                  /*      cout<<"bella zio"<<endl;
                        cout<<"bella zio"<<endl;
                        cout<<"bella zio"<<endl;
                        cout<<"bella zio"<<endl;
                        cout<<"bella zio"<<endl;
                        cout<<"bella zio"<<endl;
                        cout<<"bella zio"<<endl;
                        cout<<"bella zio"<<endl;*/
                        hid_t datasetChar = H5Dopen(fileIter, stringToChar(attributePath)); 
                        
                        
                        hsize_t* sizeData;
                        //H5Dvlen_get_buf_size(datasetChar, H5T_NATIVE_CHAR, H5S_ALL, sizeData );
                        //hid_t typeData = H5Dget_type(datasetChar);
                        //size_t sizeChar = H5Tget_size(typeData);
                        //sizeChar++;
                        //hid_t space = H5Dget_space (datasetChar);
                        //int ndims = H5Sget_simple_extent_dims (space, dims, NULL);
                        
                        //size_t attr_size = getInMemDataSize(datasetChar);
                        //bufChar = new char[(size_t)attr_size];
                        
                        
                        //bufChar = (char*)malloc((size_t)sizeData);
                        
                        //bufChar = (char*)malloc(10000);
                        
                        //H5Dread(datasetChar, H5T_NATIVE_CHAR, H5S_ALL, 
                            //          H5S_ALL, H5P_DEFAULT, bufChar);
                        //H5Dclose(datasetChar);
                        
                        hid_t space_id = H5Dget_space(datasetChar);
    
                        hssize_t num_elements = H5Sget_simple_extent_npoints(space_id);
                        
                        char bufChar[num_elements+1];
                        
                        H5LTread_dataset_char(fileIter, stringToChar(attributePath), bufChar);
                        bufChar[num_elements] = '\0';
                        
                        Icategory = charToString(bufChar);
                    }
                    else
                    {
                        
                        hid_t datasetFloat = H5Dopen(fileIter, stringToChar(attributePath)); 

                        bufFloat = (float*)malloc(sizeof(float));
                        H5Dread(datasetFloat, H5T_NATIVE_FLOAT, H5S_ALL, 
                                      H5S_ALL, H5P_DEFAULT, bufFloat);
                        H5Dclose(datasetFloat);
                    }
                    
                    
                    
                    switch(k)
                    {

                        case 1:
                            Iinstance = *bufFloat;
                            free(bufFloat);
                        break;
                        case 2:
                            Ibottom = *bufFloat;
                            free(bufFloat);
                        break;
                        case 3:
                            Itop = *bufFloat;
                            free(bufFloat);
                        break;
                        case 4:
                            Ileft = *bufFloat;
                            free(bufFloat);
                        break;
                        case 5:
                            Iright = *bufFloat;
                            free(bufFloat);
                        break;
                    }
                    
                    
                    
                }
                //aggiungo la categoria se non è già presente all'elenco delle categorie di tutto il video
                
                string category = getFormattedCategory(Icategory, Iinstance);
                bool isIn = false;
                for(int z = 0; z<tvCategories.size(); z++)
                {
                    if(tvCategories.at(z) == category)
                        isIn = true;
                }
                if(isIn == false)
                    tvCategories.push_back(category);
                    
                Item item = Item(Icategory, Iinstance, Ibottom, Itop, Ileft, Iright);
                //controllo se la ground truth è troppo piccola o troppo grande rispetto i template
                if(discardGroundTruth == false || isScaledWithGroundTruth(item))
                    vfItems.push_back(item);
                else
                    vfDiscardedItems.push_back(item);
            }
            vf.items = vfItems;
            vf.discardedItems = vfDiscardedItems;
            vf.nItem -= vfDiscardedItems.size();
            vFrames.push_back(vf);
        }
        
        tv.frames = vFrames;
        tv.categories = tvCategories;
        sort(tv.categories.begin(), tv.categories.end());
        
    }
    

    /*
    printf(" Objects in the root group are:\n");
    printf("\n");

    hid_t fileIter = H5Fopen( stringToChar(pathVideoMat), H5F_ACC_RDONLY, H5P_DEFAULT );
    H5Giterate(fileIter, "bboxes/value/_033/value/bottom/value/_0/", NULL, file_info, NULL);

    hid_t dataset = H5Dopen(fileIter, "bboxes/value/_033/value/category/value/_0/value"); 
    
    
    char * buf;
    buf = (char*)malloc(sizeof(char)*50);
    H5Dread(dataset, H5T_NATIVE_CHAR, H5S_ALL, 
                  H5S_ALL, H5P_DEFAULT, buf); 

    cout<<sizeof(buf)<<endl;
    //printf("valore: %s", *buf);
    
*/
      
    return tv;
}

bool isScaledWithGroundTruth(Item item)
{
    string class_id = getFormattedCategory(item.category, item.instance);
    int itemWidth = item.right-item.left;
    int itemHeight = item.bottom-item.top;

    int minWidth;
    int maxWidth;
    int minHeight;
    int maxHeight;
	
    string pathMeasures = "./templates/"+class_id+"/gtMeasures.yml";
    if(fileExists(pathMeasures.c_str()))
    {
        cv::FileStorage fs(pathMeasures, cv::FileStorage::READ);
        
        minWidth = (int)fs["minWidth"];
        maxWidth = (int)fs["maxWidth"];
        minHeight = (int)fs["minHeight"];
        maxHeight = (int)fs["maxHeight"];
        
        fs.release();
    } 
    
    bool accepted = true;
    int minAccept = 110;
    int maxAccept = 110;
    /*if(class_id == "soda_can_1")
    {
    cout<<"prima minW: "<<minWidth<<endl;
    cout<<"prima minH: "<<minHeight<<endl;
    cout<<"prima maxW: "<<maxWidth<<endl;
    cout<<"prima maxH: "<<maxHeight<<endl;
    }*/
    
    minWidth = (int)(((float)minWidth/100) * minAccept);
    minHeight = (int)(((float)minHeight/100) * minAccept);
    maxWidth = (int)(((float)maxWidth/100) * maxAccept);
    maxHeight = (int)(((float)maxHeight/100) * maxAccept);
    
    /*if(class_id == "soda_can_1")
    {
    cout<<"dopo minW: "<<minWidth<<endl;
    cout<<"dopo minH: "<<minHeight<<endl;
    cout<<"dopo maxW: "<<maxWidth<<endl;
    cout<<"dopo maxH: "<<maxHeight<<endl;
    }*/

    if(itemWidth <= minWidth || itemWidth >= maxWidth)
        accepted = false;
    if(itemHeight <= minHeight || itemHeight >= maxHeight)
        accepted = false;
    
    return accepted;
}

void drawGroundTruth(VideoFrame videoFrame, cv::Mat& dst)
{
    const cv::Scalar blue = CV_RGB(0,0,255);
    for(int i = 0; i<videoFrame.nItem; i++)
    {
        Item item = videoFrame.items.at(i);
        cv::Point topLeft(item.left, item.top);
        cv::Point bottomRight(item.right, item.bottom);
        rectangle(dst, topLeft, bottomRight, blue, 2);
    }
    
    
    //rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
}

bool isCategoryToCheck(vector<pair<string,string> > categoriesToCheck, string category)
{
    bool result = false;
    for(int i = 0; i<categoriesToCheck.size(); i++)
    {
        string formattedCat = categoriesToCheck.at(i).first + "_" +categoriesToCheck.at(i).second;
        if(formattedCat == category)
        {
            result = true;
            break;
        }
    }
    
    return result;
}

bool isItemToCheck(vector<pair<string,string> > categoriesToCheck, Item item)
{
    bool result = false;
    for(int i = 0; i<categoriesToCheck.size(); i++)
    {
        
        string number = intToString(item.instance);
        if(categoriesToCheck.at(i).first == item.category && categoriesToCheck.at(i).second == number)
        {
            result = true;
            break;
        }
    }
    
    return result;
}

bool isTheRightItem(Item item, my_linemod::Match m, const std::vector<cv::my_linemod::Template>& templates)
{
    Point offset = Point(m.x, m.y);
    string class_id = m.class_id;
    string formattedCategory = getFormattedCategory(item.category, item.instance);
    int num_modalities = 1;
    if(formattedCategory == class_id)
    {
        Rect_<int> Bgt = Rect_<int>(item.left, item.top, item.right-item.left, item.bottom-item.top);
        Rect_<int> Ba = Rect_<int>(offset.x, offset.y, (offset.x+templates[num_modalities-1].width - offset.x), (offset.y+templates[num_modalities-1].height - offset.y));
        Rect_<int> intersection = Bgt & Ba;
        float areaIntersection = (float)intersection.area();
        float areaUnion = (float)(Bgt.area()+Ba.area())-areaIntersection;
        float result = areaIntersection/areaUnion;
        
        if(result >= 0.4)
            return true;
        else
            return false;
    }
    else
        return false;
}

void checkFrameFalses(vector<my_linemod::Match> matches, VideoFrame videoFrame, vector<pair<string,string> > categoriesToCheck, cv::Ptr<my_linemod::Detector> detector, VideoResult& videoResult, Mat& dst, bool displayRects)
{
    
    const cv::Scalar green = CV_RGB(0,255,0);
    const cv::Scalar red = CV_RGB(255,0,0);
    const cv::Scalar purple = CV_RGB(204,51,153);
    const cv::Scalar yellow = CV_RGB(255,255,0);
    const cv::Scalar black = CV_RGB(0,0,0);
    
    int num_modalities = 1;
    int num_classes = categoriesToCheck.size();
    
    //ottengo tutte le ground truth da controllare e le metto in una mappa, per poter poi controllare se sono state rispettate o meno (falsi negativi)
    map<Item,bool> gtMap;
    map<Item,bool> ::iterator it;
    for(int i = 0; i<videoFrame.items.size(); i++)
    {
        Item item = videoFrame.items.at(i);
        if(isItemToCheck(categoriesToCheck, item))
            gtMap.insert(pair<Item,bool>(item,false));
    }

    videoResult.positivesExpected += gtMap.size();

    int classes_visited = 0;
    std::set<std::string> visited;
    
    for(int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); i++)
    {
        //se non è stato trovato almeno una gt attesa, c'è un falso positivo
        bool foundAtLeastOne = false;
        my_linemod::Match m = matches[i];
        
        if (visited.insert(m.class_id).second)
        {
            ++classes_visited;
            
            const vector<my_linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
            for ( it=gtMap.begin() ; it != gtMap.end(); it++ )
            {
                
                if(isTheRightItem((*it).first, m, templates) == true)
                {
                    
                    foundAtLeastOne = true;
                    (*it).second = true;
                }
            }
            if(foundAtLeastOne == false)
            {
                //FALSO POSITIVO
                Point offset = Point(m.x, m.y);
                Rect_<int> Ba = Rect_<int>(offset.x, offset.y, (offset.x+templates[num_modalities-1].width - offset.x), (offset.y+templates[num_modalities-1].height - offset.y));

                FalsePositive fp = FalsePositive(m.class_id, m.similarity, videoFrame.frameNumber, Ba);
                
                if(displayRects == true)
                    rectangle(dst, cv::Point(Ba.x,Ba.y), cv::Point(Ba.x+Ba.width,Ba.y+Ba.height), purple, 2);
                
                map<string, CategoryResult> ::iterator itCat;
                itCat = videoResult.mapCategoryResult.find(m.class_id);
                CategoryResult* cr = &(*itCat).second;
                cr->arrayFalsePositive.push_back(fp);
                cr->nFalsePositive++;
                videoResult.nFalsePositive++;
            }
        
        }
        
    }
    
    for ( it=gtMap.begin() ; it != gtMap.end(); it++ )
    {
        if((*it).second == false)
        {
            //FALSO NEGATIVO
            Item item = (*it).first;
            Rect_<int> Bgt = Rect_<int>(item.left, item.top, item.right-item.left, item.bottom-item.top);
            string formattedCategory = getFormattedCategory(item.category, item.instance);
            
            FalseNegative fn = FalseNegative(formattedCategory, videoFrame.frameNumber, Bgt);
            
            if(displayRects == true)
                rectangle(dst, cv::Point(Bgt.x,Bgt.y), cv::Point(Bgt.x+Bgt.width,Bgt.y+Bgt.height), red, 2);
            
            map<string, CategoryResult> ::iterator itCat;
            itCat = videoResult.mapCategoryResult.find(formattedCategory);
            CategoryResult* cr = &(*itCat).second;
            cr->arrayFalseNegative.push_back(fn);
            cr->nFalseNegative++;
            videoResult.nFalseNegative++;
        }
    }
    
    //disegno le ground truth scartate
    for(int i = 0; i<videoFrame.discardedItems.size(); i++)
    {
        Item item = videoFrame.discardedItems.at(i);
        if(isItemToCheck(categoriesToCheck, item))
        {
            Rect_<int> sctGT = Rect_<int>(item.left, item.top, item.right-item.left, item.bottom-item.top);
            if(displayRects == true)
            {
                rectangle(dst, cv::Point(sctGT.x,sctGT.y), cv::Point(sctGT.x+sctGT.width,sctGT.y+sctGT.height), black, 3);
                string text = "Width: "+intToString(sctGT.width) + " - Height: " + intToString(sctGT.height); 
                putText(dst, text, cv::Point(sctGT.x,sctGT.y-5), FONT_HERSHEY_SIMPLEX, 1, yellow); 
            }
        }
                
    }

}

bool stringCompare( const string &left, const string &right ){
   for( string::const_iterator lit = left.begin(), rit = right.begin(); lit != left.end() && rit != right.end(); ++lit, ++rit )
      if( tolower( *lit ) < tolower( *rit ) )
         return true;
      else if( tolower( *lit ) > tolower( *rit ) )
         return false;
   if( left.size() < right.size() )
      return true;
   return false;
}

bool fileExists(const char * path)
{
    FILE* fp = fopen(path, "r");
    if (fp) {
        // file exists
        fclose(fp);
        return true;
    } else {
        return false;
    }
}
void readDirectory(const char* dirname, vector<string>& listFiles)
{
    
    DIR *dir;
    struct dirent *ent;
        dir = opendir (dirname);
    if (dir != NULL) {

      /* print all the files and directories within directory */
      while ((ent = readdir (dir)) != NULL) {
        //printf ("%s\n", ent->d_name);
        listFiles.push_back(charToString(ent->d_name));
      }
      //sort (listFiles.begin(), listFiles.end(), stringCompare);
      closedir (dir);
    } else {
      /* could not open directory */
      fprintf(stderr, "ERROR: Could not open directory %s\n", dirname);
    }
   
}

void getMasksFromListFile(vector<string>& listFiles)
{
    vector<string> masks;
    for(int i = 0; i<listFiles.size(); i++)
    {
        string tempString = listFiles.at(i);
        if (tempString.find ("_mask.png") != string::npos)
        {
            masks.push_back(tempString);
            getColorFromMask(tempString);
            getDepthFromMask(tempString);
            
        }
    }
    listFiles.clear();
    listFiles = masks;
    
}

string getColorFromMask(string mask)
{
    string color;
    //size_t position = mask.find ("_mask.png");
    size_t lenght = mask.size() - 9; //9 = _mask.png
    color = mask.substr(0, lenght);
    color = color + ".png";
    
    return color;
}

string getDepthFromMask(string mask)
{
    string depth;
    //size_t position = mask.find ("_mask.png");
    size_t lenght = mask.size() - 9; //9 = _mask.png
    depth = mask.substr(0, lenght);
    depth = depth + "_depth.png";
    
    return depth;
}

void deleteResults(int nPipe, vector<int> v_threshold_rgb,vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<string> nomiVideo, vector<pair<string,string> > all_categories)
{
    bool actual_use63;
    int actual_featuresUsed;
    
    int actual_signFeat;
    
    int actual_threshold_rgb;
    
    for(int del1 = 0; del1<v_threshold_rgb.size(); del1++)
	{
		actual_threshold_rgb = v_threshold_rgb.at(del1);
	    
	for(int del2 = 0; del2<v_use63.size(); del2++)
	{
		actual_use63 = v_use63.at(del2);
		
	for(int del3 = 0; del3<v_featuresUsed.size(); del3++)
	{
	    //cambio il numero di feature solo se use63 è false
	    if(actual_use63 == false)
            actual_featuresUsed = v_featuresUsed.at(del3);
	    else //use63 è true
            actual_featuresUsed = 63;
	    
	for(int del4 = 0; del4<v_signFeat.size(); del4++)
	{
		actual_signFeat = v_signFeat.at(del4);
        
    for(int w = 0; w<all_categories.size(); w++)
    {
		pair<string,string> category = all_categories.at(w);
        string class_id = category.first + "_" + category.second;
        
        string string_use63 = "use63-" + boolToString(actual_use63);
        string string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
        string string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
        string string_signFeat = "signFeat-" + intToString(actual_signFeat);
        
        for(int test10 = 0; test10<nomiVideo.size(); test10++)
        {
            string nomeVideo;
            nomeVideo = nomiVideo.at(test10);
            for(int nm = 0; nm<nPipe; nm++)
            {
                string pipeline = "pipeline_" + intToString(nm);
                
                string pathRes = "./results/" + pipeline + "/" + nomeVideo + "/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
                if(fileExists(pathRes.c_str()))
                    remove(pathRes.c_str());
            }
        }
    }
    }  
    }  
    }
    }

}

void swapBest(pair<string, string> video_plus_category, map<pair<string, string>, bool>& map_use63, map<pair<string, string>, int>& map_featuresUsed, map<pair<string, string>, int>& map_signFeat, map<pair<string, string>, int>& map_threshold_rgb, map<pair<string, string>, bool>& map_punteggio16, map<pair<string, string>, bool>& map_featuresSignatureCandidates, map<pair<string, string>, bool>& map_signatureEnabled, map<pair<string, string>, bool>& map_grayEnabled, map<pair<string, string>, int>& map_matching, map<pair<string, string>, int>& map_expPos, map<pair<string, string>,int>& map_falsePositive, map<pair<string, string>, int>& map_falseNegative, map<pair<string, string>, int>& map_falsePosNeg,
                bool actual_use63, int actual_featuresUsed, int actual_signFeat, int actual_threshold_rgb, bool actual_punteggio16, bool actual_featuresSignatureCandidates, bool actual_grayEnabled, bool actual_signatureEnabled, int actual_matching_threshold, int actual_exp_positives, int actual_fp, int actual_fn, int actual_fpfn)
{
    map_use63[video_plus_category] = actual_use63;
    map_featuresUsed[video_plus_category] = actual_featuresUsed;
    map_signFeat[video_plus_category] = actual_signFeat;
    map_threshold_rgb[video_plus_category] = actual_threshold_rgb;
    map_punteggio16[video_plus_category] = actual_punteggio16;
    map_featuresSignatureCandidates[video_plus_category] = actual_featuresSignatureCandidates;
    map_signatureEnabled[video_plus_category] = actual_signatureEnabled;
    map_grayEnabled[video_plus_category] = actual_grayEnabled;
    map_matching[video_plus_category] = actual_matching_threshold;
    map_expPos[video_plus_category] = actual_exp_positives;
    map_falsePositive[video_plus_category] = actual_fp;
    map_falseNegative[video_plus_category] = actual_fn;
    map_falsePosNeg[video_plus_category] = actual_fpfn; 
}

void analyzeResults_category_video(int nPipe, vector<string>& nomiVideo, vector<int> v_threshold_rgb, vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<pair<string,string> > all_categories)
{
    string pipeline = "pipeline_" + intToString(nPipe);
    
    //pair< mappa<pair<video, categoria>, valore_parametro>  , primoSecondoOTerzomigliore>
    map<pair<string, string>, bool> fp_best_map_use63;
    map<pair<string, string>, int> fp_best_map_featuresUsed;
    map<pair<string, string>, int> fp_best_map_signFeat;
    map<pair<string, string>, int> fp_best_map_threshold_rgb;
    map<pair<string, string>, bool> fp_best_map_punteggio16;
    map<pair<string, string>, bool> fp_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> fp_best_map_signatureEnabled;
    map<pair<string, string>, bool> fp_best_map_grayEnabled;
    map<pair<string, string>, int> fp_best_map_matching;
    map<pair<string, string>, int> fp_best_map_expectedPositive;
    map<pair<string, string>, int> fp_best_map_falsePositive;
    map<pair<string, string>, int> fp_best_map_falseNegative;
    map<pair<string, string>, int> fp_best_map_falsePosNeg;    
    
    map<pair<string, string>, bool> second_fp_best_map_use63;
    map<pair<string, string>, int> second_fp_best_map_featuresUsed;
    map<pair<string, string>, int> second_fp_best_map_signFeat;
    map<pair<string, string>, int> second_fp_best_map_threshold_rgb;
    map<pair<string, string>, bool> second_fp_best_map_punteggio16;
    map<pair<string, string>, bool> second_fp_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> second_fp_best_map_signatureEnabled;
    map<pair<string, string>, bool> second_fp_best_map_grayEnabled;
    map<pair<string, string>, int> second_fp_best_map_matching;
    map<pair<string, string>, int> second_fp_best_map_expectedPositive;
    map<pair<string, string>, int> second_fp_best_map_falsePositive;
    map<pair<string, string>, int> second_fp_best_map_falseNegative;
    map<pair<string, string>, int> second_fp_best_map_falsePosNeg;  
    
    map<pair<string, string>, bool> third_fp_best_map_use63;
    map<pair<string, string>, int> third_fp_best_map_featuresUsed;
    map<pair<string, string>, int> third_fp_best_map_signFeat;
    map<pair<string, string>, int> third_fp_best_map_threshold_rgb;
    map<pair<string, string>, bool> third_fp_best_map_punteggio16;
    map<pair<string, string>, bool> third_fp_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> third_fp_best_map_signatureEnabled;
    map<pair<string, string>, bool> third_fp_best_map_grayEnabled;
    map<pair<string, string>, int> third_fp_best_map_matching;
    map<pair<string, string>, int> third_fp_best_map_expectedPositive;
    map<pair<string, string>, int> third_fp_best_map_falsePositive;
    map<pair<string, string>, int> third_fp_best_map_falseNegative;
    map<pair<string, string>, int> third_fp_best_map_falsePosNeg; 
    
    map<pair<string, string>, bool> fn_best_map_use63;
    map<pair<string, string>, int> fn_best_map_featuresUsed;
    map<pair<string, string>, int> fn_best_map_signFeat;
    map<pair<string, string>, int> fn_best_map_threshold_rgb;
    map<pair<string, string>, bool> fn_best_map_punteggio16;
    map<pair<string, string>, bool> fn_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> fn_best_map_signatureEnabled;
    map<pair<string, string>, bool> fn_best_map_grayEnabled;
    map<pair<string, string>, int> fn_best_map_matching;
    map<pair<string, string>, int> fn_best_map_expectedPositive;
    map<pair<string, string>, int> fn_best_map_falsePositive;
    map<pair<string, string>, int> fn_best_map_falseNegative;
    map<pair<string, string>, int> fn_best_map_falsePosNeg;    
    
    map<pair<string, string>, bool> second_fn_best_map_use63;
    map<pair<string, string>, int> second_fn_best_map_featuresUsed;
    map<pair<string, string>, int> second_fn_best_map_signFeat;
    map<pair<string, string>, int> second_fn_best_map_threshold_rgb;
    map<pair<string, string>, bool> second_fn_best_map_punteggio16;
    map<pair<string, string>, bool> second_fn_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> second_fn_best_map_signatureEnabled;
    map<pair<string, string>, bool> second_fn_best_map_grayEnabled;
    map<pair<string, string>, int> second_fn_best_map_matching;
    map<pair<string, string>, int> second_fn_best_map_expectedPositive;
    map<pair<string, string>, int> second_fn_best_map_falsePositive;
    map<pair<string, string>, int> second_fn_best_map_falseNegative;
    map<pair<string, string>, int> second_fn_best_map_falsePosNeg;   
    
    map<pair<string, string>, bool> third_fn_best_map_use63;
    map<pair<string, string>, int> third_fn_best_map_featuresUsed;
    map<pair<string, string>, int> third_fn_best_map_signFeat;
    map<pair<string, string>, int> third_fn_best_map_threshold_rgb;
    map<pair<string, string>, bool> third_fn_best_map_punteggio16;
    map<pair<string, string>, bool> third_fn_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> third_fn_best_map_signatureEnabled;
    map<pair<string, string>, bool> third_fn_best_map_grayEnabled;
    map<pair<string, string>, int> third_fn_best_map_matching;
    map<pair<string, string>, int> third_fn_best_map_expectedPositive;
    map<pair<string, string>, int> third_fn_best_map_falsePositive;
    map<pair<string, string>, int> third_fn_best_map_falseNegative;
    map<pair<string, string>, int> third_fn_best_map_falsePosNeg;   
    
    map<pair<string, string>, bool> fpfn_best_map_use63;
    map<pair<string, string>, int> fpfn_best_map_featuresUsed;
    map<pair<string, string>, int> fpfn_best_map_signFeat;
    map<pair<string, string>, int> fpfn_best_map_threshold_rgb;
    map<pair<string, string>, bool> fpfn_best_map_punteggio16;
    map<pair<string, string>, bool> fpfn_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> fpfn_best_map_signatureEnabled;
    map<pair<string, string>, bool> fpfn_best_map_grayEnabled;
    map<pair<string, string>, int> fpfn_best_map_matching;
    map<pair<string, string>, int> fpfn_best_map_expectedPositive;
    map<pair<string, string>, int> fpfn_best_map_falsePositive;
    map<pair<string, string>, int> fpfn_best_map_falseNegative;
    map<pair<string, string>, int> fpfn_best_map_falsePosNeg;    
    
    map<pair<string, string>, bool> second_fpfn_best_map_use63;
    map<pair<string, string>, int> second_fpfn_best_map_featuresUsed;
    map<pair<string, string>, int> second_fpfn_best_map_signFeat;
    map<pair<string, string>, int> second_fpfn_best_map_threshold_rgb;
    map<pair<string, string>, bool> second_fpfn_best_map_punteggio16;
    map<pair<string, string>, bool> second_fpfn_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> second_fpfn_best_map_signatureEnabled;
    map<pair<string, string>, bool> second_fpfn_best_map_grayEnabled;
    map<pair<string, string>, int> second_fpfn_best_map_matching;
    map<pair<string, string>, int> second_fpfn_best_map_expectedPositive;
    map<pair<string, string>, int> second_fpfn_best_map_falsePositive;
    map<pair<string, string>, int> second_fpfn_best_map_falseNegative;
    map<pair<string, string>, int> second_fpfn_best_map_falsePosNeg;    
    
    map<pair<string, string>, bool> third_fpfn_best_map_use63;
    map<pair<string, string>, int> third_fpfn_best_map_featuresUsed;
    map<pair<string, string>, int> third_fpfn_best_map_signFeat;
    map<pair<string, string>, int> third_fpfn_best_map_threshold_rgb;
    map<pair<string, string>, bool> third_fpfn_best_map_punteggio16;
    map<pair<string, string>, bool> third_fpfn_best_map_featuresSignatureCandidates;
    map<pair<string, string>, bool> third_fpfn_best_map_signatureEnabled;
    map<pair<string, string>, bool> third_fpfn_best_map_grayEnabled;
    map<pair<string, string>, int> third_fpfn_best_map_matching;
    map<pair<string, string>, int> third_fpfn_best_map_expectedPositive;
    map<pair<string, string>, int> third_fpfn_best_map_falsePositive;
    map<pair<string, string>, int> third_fpfn_best_map_falseNegative;
    map<pair<string, string>, int> third_fpfn_best_map_falsePosNeg;    
    
    
    bool actual_use63;
    int actual_featuresUsed;
    
    int actual_signFeat;
    
    int actual_threshold_rgb;
    
    for(int del1 = 0; del1<v_threshold_rgb.size(); del1++)
	{
		actual_threshold_rgb = v_threshold_rgb.at(del1);
	    
	for(int del2 = 0; del2<v_use63.size(); del2++)
	{
		actual_use63 = v_use63.at(del2);
		
	for(int del3 = 0; del3<v_featuresUsed.size(); del3++)
	{
	    //cambio il numero di feature solo se use63 è false
	    if(actual_use63 == false)
            actual_featuresUsed = v_featuresUsed.at(del3);
	    else //use63 è true
            actual_featuresUsed = 63;
	    
	for(int del4 = 0; del4<v_signFeat.size(); del4++)
	{
		actual_signFeat = v_signFeat.at(del4);
    
    for(int w = 0; w<all_categories.size(); w++)
    {
        cout<<"."<<flush;
		pair<string,string> category = all_categories.at(w);
        string class_id = category.first + "_" + category.second;
        
        string string_use63 = "use63-" + boolToString(actual_use63);
        string string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
        string string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
        string string_signFeat = "signFeat-" + intToString(actual_signFeat);
        
        for(int test10 = 0; test10<nomiVideo.size(); test10++)
        {
            string nomeVideo;
            nomeVideo = nomiVideo.at(test10);
        
            
            
            string pathRes = "./results/" + pipeline + "/" + nomeVideo + "/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
            if(fileExists(pathRes.c_str()))
            {
                cv::FileStorage fs(pathRes, cv::FileStorage::READ);
        
                vector<bool> vr_punteggio16;
                vector<bool> vr_featuresSignatureCandidates;
                vector<bool> vr_signatureEnabled;
                vector<bool> vr_grayEnabled;
                vector<int> vr_matching_threshold;
                vector<int> vr_exp_positives;
                vector<int> vr_nFalsePositives;
                vector<int> vr_nFalseNegatives;
                
                cv::FileNode fn = fs["tests"];
                for (cv::FileNodeIterator i = fn.begin(); i != fn.end(); ++i)
                {
                    vr_punteggio16.push_back(((int)(*i)["punteggio16"]));
                    vr_featuresSignatureCandidates.push_back(((int)(*i)["featuresSignCand"]));
                    vr_signatureEnabled.push_back(((int)(*i)["signatureEnabled"]));
                    vr_grayEnabled.push_back(((int)(*i)["grayEnabled"]));
                    vr_matching_threshold.push_back((*i)["matching"]);
                    vr_exp_positives.push_back((*i)["Expected positives"]);
                    vr_nFalsePositives.push_back((*i)["False Positives"]);
                    vr_nFalseNegatives.push_back((*i)["False Negatives"]);
                }
                
                fs.release();
                
                

                for(int res = 0; res<vr_punteggio16.size(); res++)
                {
                    //riempimento mappe dei miglior risultati
                    //le inizializzo al massimo valore possibile per poter poi sostituire con i migliori
                    pair<string, string> pair_result = make_pair(nomeVideo, class_id);           
                    int maxInt = std::numeric_limits<int>::max();    
                    int tmp_best_fp = maxInt;
                    int tmp_best_fn = maxInt;
                    int tmp_best_fpfn = maxInt;
                    int tmp_second_best_fp = maxInt;
                    int tmp_second_best_fn = maxInt;
                    int tmp_second_best_fpfn = maxInt;
                    int tmp_third_best_fp = maxInt;
                    int tmp_third_best_fn = maxInt;
                    int tmp_third_best_fpfn = maxInt;
                    
                    //miglior falsi positivi attuale
                    map<pair<string, string>, int>::iterator it_best_fp = fp_best_map_falsePositive.find(pair_result);
                    if(it_best_fp != fp_best_map_falsePositive.end())
                        tmp_best_fp = (*it_best_fp).second;
                    //miglior falsi negativi attuale
                    map<pair<string, string>, int>::iterator it_best_fn = fn_best_map_falseNegative.find(pair_result);
                    if(it_best_fn != fn_best_map_falseNegative.end())
                        tmp_best_fn = (*it_best_fn).second;
                    //miglior somma falsi positivi/negativi attuale
                    map<pair<string, string>, int>::iterator it_best_fpfn = fpfn_best_map_falsePosNeg.find(pair_result);
                    if(it_best_fpfn != fpfn_best_map_falsePosNeg.end())
                        tmp_best_fpfn = (*it_best_fpfn).second;
                        
                    //secondo miglior falsi positivi attuale
                    map<pair<string, string>, int>::iterator it_second_best_fp = second_fp_best_map_falsePositive.find(pair_result);
                    if(it_second_best_fp != second_fp_best_map_falsePositive.end())
                        tmp_second_best_fp = (*it_second_best_fp).second;
                    //secondo falsi negativi attuale
                    map<pair<string, string>, int>::iterator it_second_best_fn = second_fn_best_map_falseNegative.find(pair_result);
                    if(it_second_best_fn != second_fn_best_map_falseNegative.end())
                        tmp_second_best_fn = (*it_second_best_fn).second;
                    //secondo somma falsi positivi/negativi attuale
                    map<pair<string, string>, int>::iterator it_second_best_fpfn = second_fpfn_best_map_falsePosNeg.find(pair_result);
                    if(it_second_best_fpfn != second_fpfn_best_map_falsePosNeg.end())
                        tmp_second_best_fpfn = (*it_second_best_fpfn).second;
                        
                        
                    //terzo miglior falsi positivi attuale
                    map<pair<string, string>, int>::iterator it_third_best_fp = third_fp_best_map_falsePositive.find(pair_result);
                    if(it_third_best_fp != third_fp_best_map_falsePositive.end())
                        tmp_third_best_fp = (*it_third_best_fp).second;
                    //terzo falsi negativi attuale
                    map<pair<string, string>, int>::iterator it_third_best_fn = third_fn_best_map_falseNegative.find(pair_result);
                    if(it_third_best_fn != third_fn_best_map_falseNegative.end())
                        tmp_third_best_fn = (*it_third_best_fn).second;
                    //terzo somma falsi positivi/negativi attuale
                    map<pair<string, string>, int>::iterator it_third_best_fpfn = third_fpfn_best_map_falsePosNeg.find(pair_result);
                    if(it_third_best_fpfn != third_fpfn_best_map_falsePosNeg.end())
                        tmp_third_best_fpfn = (*it_third_best_fpfn).second;
                        
                    int actual_exp_positives = vr_exp_positives.at(res);
                    int actual_fp = vr_nFalsePositives.at(res);
                    int actual_fn = vr_nFalseNegatives.at(res);
                    int actual_fpfn = actual_fp + actual_fn;
                    
                    /////////AGGIORNO I MIGLIORI FALSI POSITIVI//////////
                    
                    bool tmpWithFirst = false;
                    bool tmpWithSecond = false;
                    bool tmpWithThird = false;
                    bool firstWithSecond = false;
                    bool secondWithThird = false;
                    
                    //se il miglior risultato è peggiore dell'attuale
                    //PEGGIORE == MAGGIORE
                    if(tmp_best_fp > actual_fp)
                    {
                        //cambio il best con l'attuale
                        tmpWithFirst = true;
                        //il primo con il secondo 
                        if(tmp_best_fp < maxInt) //se = a maxInt, ancora non è stato inizializzato, quindi non dovrò passare i valori al secondo e terzo
                            firstWithSecond = true;
                        //il secondo con il terzo
                        if(tmp_second_best_fp < maxInt)
                            secondWithThird = true;
                    }
                    else
                    {
                        //prova a cambiare tmp con il secondo
                        //se il secondo è peggiore, cambia tmp con il secondo
                        if(tmp_second_best_fp > actual_fp)
                        {
                            tmpWithSecond = true;
                            //il secondo con il terzo
                            if(tmp_second_best_fp < maxInt)
                                secondWithThird = true;
                        }
                        else
                        {
                            //altrimenti se il secondo è migliore, prova a cambiare tmp con il terzo
                            //se il terzo è peggiore cambialo - FINITO
                            if(tmp_third_best_fp > actual_fp)
                                tmpWithThird = true;
                        }
                    }
                    
                    if(secondWithThird == true)
                        swapBest(pair_result, third_fp_best_map_use63, third_fp_best_map_featuresUsed, third_fp_best_map_signFeat, third_fp_best_map_threshold_rgb, third_fp_best_map_punteggio16, third_fp_best_map_featuresSignatureCandidates, third_fp_best_map_signatureEnabled, third_fp_best_map_grayEnabled, third_fp_best_map_matching, third_fp_best_map_expectedPositive, third_fp_best_map_falsePositive, third_fp_best_map_falseNegative, third_fp_best_map_falsePosNeg,
                                    second_fp_best_map_use63[pair_result], second_fp_best_map_featuresUsed[pair_result], second_fp_best_map_signFeat[pair_result], second_fp_best_map_threshold_rgb[pair_result], second_fp_best_map_punteggio16[pair_result], second_fp_best_map_featuresSignatureCandidates[pair_result], second_fp_best_map_grayEnabled[pair_result], second_fp_best_map_signatureEnabled[pair_result], second_fp_best_map_matching[pair_result], second_fp_best_map_expectedPositive[pair_result], second_fp_best_map_falsePositive[pair_result], second_fp_best_map_falseNegative[pair_result], second_fp_best_map_falsePosNeg[pair_result]);
                                    
                    if(firstWithSecond == true)
                        swapBest(pair_result, second_fp_best_map_use63, second_fp_best_map_featuresUsed, second_fp_best_map_signFeat, second_fp_best_map_threshold_rgb, second_fp_best_map_punteggio16, second_fp_best_map_featuresSignatureCandidates, second_fp_best_map_signatureEnabled, second_fp_best_map_grayEnabled, second_fp_best_map_matching, second_fp_best_map_expectedPositive ,second_fp_best_map_falsePositive, second_fp_best_map_falseNegative, second_fp_best_map_falsePosNeg,
                                    fp_best_map_use63[pair_result], fp_best_map_featuresUsed[pair_result], fp_best_map_signFeat[pair_result], fp_best_map_threshold_rgb[pair_result], fp_best_map_punteggio16[pair_result], fp_best_map_featuresSignatureCandidates[pair_result], fp_best_map_grayEnabled[pair_result], fp_best_map_signatureEnabled[pair_result], fp_best_map_matching[pair_result], fp_best_map_expectedPositive[pair_result], fp_best_map_falsePositive[pair_result], fp_best_map_falseNegative[pair_result], fp_best_map_falsePosNeg[pair_result]);
                    
                    if(tmpWithThird == true)
                        swapBest(pair_result, third_fp_best_map_use63, third_fp_best_map_featuresUsed, third_fp_best_map_signFeat, third_fp_best_map_threshold_rgb, third_fp_best_map_punteggio16, third_fp_best_map_featuresSignatureCandidates, third_fp_best_map_signatureEnabled, third_fp_best_map_grayEnabled, third_fp_best_map_matching, third_fp_best_map_expectedPositive, third_fp_best_map_falsePositive, third_fp_best_map_falseNegative, third_fp_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                    
                    if(tmpWithSecond == true)
                        swapBest(pair_result, second_fp_best_map_use63, second_fp_best_map_featuresUsed, second_fp_best_map_signFeat, second_fp_best_map_threshold_rgb, second_fp_best_map_punteggio16, second_fp_best_map_featuresSignatureCandidates, second_fp_best_map_signatureEnabled, second_fp_best_map_grayEnabled, second_fp_best_map_matching, second_fp_best_map_expectedPositive, second_fp_best_map_falsePositive, second_fp_best_map_falseNegative, second_fp_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                    
                    if(tmpWithFirst == true)
                        swapBest(pair_result, fp_best_map_use63, fp_best_map_featuresUsed, fp_best_map_signFeat, fp_best_map_threshold_rgb, fp_best_map_punteggio16, fp_best_map_featuresSignatureCandidates, fp_best_map_signatureEnabled, fp_best_map_grayEnabled, fp_best_map_matching, fp_best_map_expectedPositive, fp_best_map_falsePositive, fp_best_map_falseNegative, fp_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                        
                    //////////////////////////////////
                    
                    /////////AGGIORNO I MIGLIORI FALSI NEGATIVI//////////////////
                    
                    tmpWithFirst = false;
                    tmpWithSecond = false;
                    tmpWithThird = false;
                    firstWithSecond = false;
                    secondWithThird = false;
                    
                    //se il miglior risultato è peggiore dell'attuale
                    //PEGGIORE == MAGGIORE
                    if(tmp_best_fn > actual_fn)
                    {
                        //cambio il best con l'attuale
                        tmpWithFirst = true;
                        //il primo con il secondo
                        if(tmp_best_fn < maxInt)
                            firstWithSecond = true;
                        //il secondo con il terzo
                        if(tmp_second_best_fn < maxInt)
                            secondWithThird = true;
                    }
                    else
                    {
                        //prova a cambiare tmp con il secondo
                        //se il secondo è peggiore, cambia tmp con il secondo
                        if(tmp_second_best_fn > actual_fn)
                        {
                            tmpWithSecond = true;
                            //il secondo con il terzo
                            if(tmp_second_best_fn < maxInt)
                                secondWithThird = true;
                        }
                        else
                        {
                            //altrimenti se il secondo è migliore, prova a cambiare tmp con il terzo
                            //se il terzo è peggiore cambialo - FINITO
                            if(tmp_third_best_fn > actual_fn)
                                tmpWithThird = true;
                        }
                    }
                    
                    if(secondWithThird == true)
                        swapBest(pair_result, third_fn_best_map_use63, third_fn_best_map_featuresUsed, third_fn_best_map_signFeat, third_fn_best_map_threshold_rgb, third_fn_best_map_punteggio16, third_fn_best_map_featuresSignatureCandidates, third_fn_best_map_signatureEnabled, third_fn_best_map_grayEnabled, third_fn_best_map_matching, third_fn_best_map_expectedPositive, third_fn_best_map_falsePositive, third_fn_best_map_falseNegative, third_fn_best_map_falsePosNeg,
                                    second_fn_best_map_use63[pair_result], second_fn_best_map_featuresUsed[pair_result], second_fn_best_map_signFeat[pair_result], second_fn_best_map_threshold_rgb[pair_result], second_fn_best_map_punteggio16[pair_result], second_fn_best_map_featuresSignatureCandidates[pair_result], second_fn_best_map_grayEnabled[pair_result], second_fn_best_map_signatureEnabled[pair_result], second_fn_best_map_matching[pair_result], second_fn_best_map_expectedPositive[pair_result], second_fn_best_map_falsePositive[pair_result], second_fn_best_map_falseNegative[pair_result], second_fn_best_map_falsePosNeg[pair_result]);
                                    
                    if(firstWithSecond == true)
                        swapBest(pair_result, second_fn_best_map_use63, second_fn_best_map_featuresUsed, second_fn_best_map_signFeat, second_fn_best_map_threshold_rgb, second_fn_best_map_punteggio16, second_fn_best_map_featuresSignatureCandidates, second_fn_best_map_signatureEnabled, second_fn_best_map_grayEnabled, second_fn_best_map_matching, second_fn_best_map_expectedPositive, second_fn_best_map_falsePositive, second_fn_best_map_falseNegative, second_fn_best_map_falsePosNeg,
                                    fn_best_map_use63[pair_result], fn_best_map_featuresUsed[pair_result], fn_best_map_signFeat[pair_result], fn_best_map_threshold_rgb[pair_result], fn_best_map_punteggio16[pair_result], fn_best_map_featuresSignatureCandidates[pair_result], fn_best_map_grayEnabled[pair_result], fn_best_map_signatureEnabled[pair_result], fn_best_map_matching[pair_result], fn_best_map_expectedPositive[pair_result], fn_best_map_falsePositive[pair_result], fn_best_map_falseNegative[pair_result], fn_best_map_falsePosNeg[pair_result]);
                    
                    if(tmpWithThird == true)
                        swapBest(pair_result, third_fn_best_map_use63, third_fn_best_map_featuresUsed, third_fn_best_map_signFeat, third_fn_best_map_threshold_rgb, third_fn_best_map_punteggio16, third_fn_best_map_featuresSignatureCandidates, third_fn_best_map_signatureEnabled, third_fn_best_map_grayEnabled, third_fn_best_map_matching, third_fn_best_map_expectedPositive, third_fn_best_map_falsePositive, third_fn_best_map_falseNegative, third_fn_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                    
                    if(tmpWithSecond == true)
                        swapBest(pair_result, second_fn_best_map_use63, second_fn_best_map_featuresUsed, second_fn_best_map_signFeat, second_fn_best_map_threshold_rgb, second_fn_best_map_punteggio16, second_fn_best_map_featuresSignatureCandidates, second_fn_best_map_signatureEnabled, second_fn_best_map_grayEnabled, second_fn_best_map_matching, second_fn_best_map_expectedPositive, second_fn_best_map_falsePositive, second_fn_best_map_falseNegative, second_fn_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                    
                    if(tmpWithFirst == true)
                        swapBest(pair_result, fn_best_map_use63, fn_best_map_featuresUsed, fn_best_map_signFeat, fn_best_map_threshold_rgb, fn_best_map_punteggio16, fn_best_map_featuresSignatureCandidates, fn_best_map_signatureEnabled, fn_best_map_grayEnabled, fn_best_map_matching, fn_best_map_expectedPositive, fn_best_map_falsePositive, fn_best_map_falseNegative, fn_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                    
                    ///////////////////////////////////////////////
                    
                    /////////AGGIORNO LA MIGLIORE SOMMA DI FALSI POSITIVI+NEGATIVI//////////////////
                    
                    tmpWithFirst = false;
                    tmpWithSecond = false;
                    tmpWithThird = false;
                    firstWithSecond = false;
                    secondWithThird = false;
                    
                    //se il miglior risultato è peggiore dell'attuale
                    //PEGGIORE == MAGGIORE
                    if(tmp_best_fpfn > actual_fpfn)
                    {
                        //cambio il best con l'attuale
                        tmpWithFirst = true;
                        //il primo con il secondo
                        if(tmp_best_fpfn < maxInt)
                            firstWithSecond = true;
                        //il secondo con il terzo
                        if(tmp_second_best_fpfn < maxInt)
                            secondWithThird = true;
                    }
                    else
                    {
                        //prova a cambiare tmp con il secondo
                        //se il secondo è peggiore, cambia tmp con il secondo
                        if(tmp_second_best_fpfn > actual_fpfn)
                        {
                            tmpWithSecond = true;
                            //il secondo con il terzo
                            if(tmp_second_best_fpfn < maxInt)
                                secondWithThird = true;
                        }
                        else
                        {
                            //altrimenti se il secondo è migliore, prova a cambiare tmp con il terzo
                            //se il terzo è peggiore cambialo - FINITO
                            if(tmp_third_best_fpfn > actual_fpfn)
                                tmpWithThird = true;
                        }
                    }
                    
                    if(secondWithThird == true)
                        swapBest(pair_result, third_fpfn_best_map_use63, third_fpfn_best_map_featuresUsed, third_fpfn_best_map_signFeat, third_fpfn_best_map_threshold_rgb, third_fpfn_best_map_punteggio16, third_fpfn_best_map_featuresSignatureCandidates, third_fpfn_best_map_signatureEnabled, third_fpfn_best_map_grayEnabled, third_fpfn_best_map_matching, third_fpfn_best_map_expectedPositive, third_fpfn_best_map_falsePositive, third_fpfn_best_map_falseNegative, third_fpfn_best_map_falsePosNeg,
                                    second_fpfn_best_map_use63[pair_result], second_fpfn_best_map_featuresUsed[pair_result], second_fpfn_best_map_signFeat[pair_result], second_fpfn_best_map_threshold_rgb[pair_result], second_fpfn_best_map_punteggio16[pair_result], second_fpfn_best_map_featuresSignatureCandidates[pair_result], second_fpfn_best_map_grayEnabled[pair_result], second_fpfn_best_map_signatureEnabled[pair_result], second_fpfn_best_map_matching[pair_result], second_fpfn_best_map_expectedPositive[pair_result], second_fpfn_best_map_falsePositive[pair_result], second_fpfn_best_map_falseNegative[pair_result], second_fpfn_best_map_falsePosNeg[pair_result]);
                                    
                    if(firstWithSecond == true)
                        swapBest(pair_result, second_fpfn_best_map_use63, second_fpfn_best_map_featuresUsed, second_fpfn_best_map_signFeat, second_fpfn_best_map_threshold_rgb, second_fpfn_best_map_punteggio16, second_fpfn_best_map_featuresSignatureCandidates, second_fpfn_best_map_signatureEnabled, second_fpfn_best_map_grayEnabled, second_fpfn_best_map_matching, second_fpfn_best_map_expectedPositive, second_fpfn_best_map_falsePositive, second_fpfn_best_map_falseNegative, second_fpfn_best_map_falsePosNeg,
                                    fpfn_best_map_use63[pair_result], fpfn_best_map_featuresUsed[pair_result], fpfn_best_map_signFeat[pair_result], fpfn_best_map_threshold_rgb[pair_result], fpfn_best_map_punteggio16[pair_result], fpfn_best_map_featuresSignatureCandidates[pair_result], fpfn_best_map_grayEnabled[pair_result], fpfn_best_map_signatureEnabled[pair_result], fpfn_best_map_matching[pair_result], fpfn_best_map_expectedPositive[pair_result], fpfn_best_map_falsePositive[pair_result], fpfn_best_map_falseNegative[pair_result], fpfn_best_map_falsePosNeg[pair_result]);
                    
                    if(tmpWithThird == true)
                        swapBest(pair_result, third_fpfn_best_map_use63, third_fpfn_best_map_featuresUsed, third_fpfn_best_map_signFeat, third_fpfn_best_map_threshold_rgb, third_fpfn_best_map_punteggio16, third_fpfn_best_map_featuresSignatureCandidates, third_fpfn_best_map_signatureEnabled, third_fpfn_best_map_grayEnabled, third_fpfn_best_map_matching, third_fpfn_best_map_expectedPositive, third_fpfn_best_map_falsePositive, third_fpfn_best_map_falseNegative, third_fpfn_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                    
                    if(tmpWithSecond == true)
                        swapBest(pair_result, second_fpfn_best_map_use63, second_fpfn_best_map_featuresUsed, second_fpfn_best_map_signFeat, second_fpfn_best_map_threshold_rgb, second_fpfn_best_map_punteggio16, second_fpfn_best_map_featuresSignatureCandidates, second_fpfn_best_map_signatureEnabled, second_fpfn_best_map_grayEnabled, second_fpfn_best_map_matching, second_fpfn_best_map_expectedPositive, second_fpfn_best_map_falsePositive, second_fpfn_best_map_falseNegative, second_fpfn_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                    
                    if(tmpWithFirst == true)
                        swapBest(pair_result, fpfn_best_map_use63, fpfn_best_map_featuresUsed, fpfn_best_map_signFeat, fpfn_best_map_threshold_rgb, fpfn_best_map_punteggio16, fpfn_best_map_featuresSignatureCandidates, fpfn_best_map_signatureEnabled, fpfn_best_map_grayEnabled, fpfn_best_map_matching, fpfn_best_map_expectedPositive, fpfn_best_map_falsePositive, fpfn_best_map_falseNegative, fpfn_best_map_falsePosNeg,
                                    actual_use63, actual_featuresUsed, actual_signFeat, actual_threshold_rgb, vr_punteggio16.at(res), vr_featuresSignatureCandidates.at(res), vr_grayEnabled.at(res), vr_signatureEnabled.at(res), vr_matching_threshold.at(res), actual_exp_positives, actual_fp, actual_fn, actual_fpfn);
                
                
                }

            }
        }
    }
    cout<<endl;
    }  
    }  
    }
    }
    
    //////////////////////////////////////////////
    map<pair<string, string>, bool>::iterator it = fp_best_map_use63.begin();
    for(; it != fp_best_map_use63.end(); ++it)
    {
        pair<string, string> actual_pair = (*it).first;
        string nomeVideo = actual_pair.first;
        string class_id = actual_pair.second;
        
        string dirVideoCat = "./results/" + pipeline + "/" + nomeVideo + "/" + class_id + "/" + "BEST_RESULTS.yml";
        
        cv::FileStorage fs_best(dirVideoCat, cv::FileStorage::WRITE);
        fs_best << "BEST FALSE POSITIVES" << "[";
        
            fs_best << "{";
                fs_best << "position" << 1;
                fs_best << "use63" << fp_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << fp_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << fp_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << fp_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << fp_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << fp_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << fp_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << fp_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << fp_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<fp_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<fp_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<fp_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<fp_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 2;
                fs_best << "use63" << second_fp_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << second_fp_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << second_fp_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << second_fp_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << second_fp_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << second_fp_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << second_fp_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << second_fp_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << second_fp_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<second_fp_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<second_fp_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<second_fp_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<second_fp_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 3;
                fs_best << "use63" << third_fp_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << third_fp_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << third_fp_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << third_fp_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << third_fp_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << third_fp_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << third_fp_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << third_fp_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << third_fp_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<third_fp_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<third_fp_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<third_fp_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<third_fp_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
        fs_best << "]";
        fs_best << "BEST FALSE NEGATIVES" << "[";
        
            fs_best << "{";
                fs_best << "position" << 1;
                fs_best << "use63" << fn_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << fn_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << fn_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << fn_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << fn_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << fn_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << fn_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << fn_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << fn_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<fn_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<fn_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<fn_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<fn_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 2;
                fs_best << "use63" << second_fn_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << second_fn_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << second_fn_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << second_fn_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << second_fn_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << second_fn_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << second_fn_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << second_fn_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << second_fn_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<second_fn_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<second_fn_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<second_fn_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<second_fn_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 3;
                fs_best << "use63" << third_fn_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << third_fn_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << third_fn_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << third_fn_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << third_fn_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << third_fn_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << third_fn_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << third_fn_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << third_fn_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<third_fn_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<third_fn_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<third_fn_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<third_fn_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
        fs_best << "]";
        fs_best << "BEST FALSE SUM" << "[";
            fs_best << "{";
                fs_best << "position" << 1;
                fs_best << "use63" << fpfn_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << fpfn_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << fpfn_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << fpfn_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << fpfn_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << fpfn_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << fpfn_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << fpfn_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << fpfn_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<fpfn_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<fpfn_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<fpfn_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<fpfn_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 2;
                fs_best << "use63" << second_fpfn_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << second_fpfn_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << second_fpfn_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << second_fpfn_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << second_fpfn_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << second_fpfn_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << second_fpfn_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << second_fpfn_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << second_fpfn_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<second_fpfn_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<second_fpfn_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<second_fpfn_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<second_fpfn_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 3;
                fs_best << "use63" << third_fpfn_best_map_use63[actual_pair];
                fs_best << "featuresUsed" << third_fpfn_best_map_featuresUsed[actual_pair];
                fs_best << "signFeat" << third_fpfn_best_map_signFeat[actual_pair];
                fs_best << "threshold_rgb" << third_fpfn_best_map_threshold_rgb[actual_pair];
                fs_best << "signatureEnabled" << third_fpfn_best_map_signatureEnabled[actual_pair];
                fs_best << "grayEnabled" << third_fpfn_best_map_grayEnabled[actual_pair];
                fs_best << "featuresSignCand" << third_fpfn_best_map_featuresSignatureCandidates[actual_pair];
                fs_best << "matching"  << third_fpfn_best_map_matching[actual_pair];
                fs_best << "punteggio16"  << third_fpfn_best_map_punteggio16[actual_pair];
                fs_best << "results" << "-----";
                fs_best <<"Expected positives"<<third_fpfn_best_map_expectedPositive[actual_pair];
                fs_best <<"False Positives"<<third_fpfn_best_map_falsePositive[actual_pair];
                fs_best <<"False Negatives"<<third_fpfn_best_map_falseNegative[actual_pair];
                fs_best <<"FN_plus_FP"<<third_fpfn_best_map_falsePosNeg[actual_pair];
            fs_best << "}";
        fs_best << "]";
        
    } 
    
    
    
}

void analyzeResults_video(int nPipe, vector<string>& nomiVideo, vector<int> v_threshold_rgb, vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<pair<string,string> > all_categories)
{
    string pipeline = "pipeline_" + intToString(nPipe);
    
    //< mappa<pair<video, parametriUsati>, valore_parametro>
    map<pair<string, string>, int> map_expectedPositive;
    map<pair<string, string>, int> map_falsePositive;
    map<pair<string, string>, int> map_falseNegative;
    map<pair<string, string>, int> map_falsePosNeg;    
    
    
    bool actual_use63;
    int actual_featuresUsed;
    
    int actual_signFeat;
    
    int actual_threshold_rgb;
    
    for(int del1 = 0; del1<v_threshold_rgb.size(); del1++)
	{
		actual_threshold_rgb = v_threshold_rgb.at(del1);
	    
	for(int del2 = 0; del2<v_use63.size(); del2++)
	{
		actual_use63 = v_use63.at(del2);
		
	for(int del3 = 0; del3<v_featuresUsed.size(); del3++)
	{
	    //cambio il numero di feature solo se use63 è false
	    if(actual_use63 == false)
            actual_featuresUsed = v_featuresUsed.at(del3);
	    else //use63 è true
            actual_featuresUsed = 63;
	    
	for(int del4 = 0; del4<v_signFeat.size(); del4++)
	{
		actual_signFeat = v_signFeat.at(del4);
    
    for(int w = 0; w<all_categories.size(); w++)
    {
        cout<<"."<<flush;
		pair<string,string> category = all_categories.at(w);
        string class_id = category.first + "_" + category.second;
        
        string string_use63 = "use63-" + boolToString(actual_use63);
        string string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
        string string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
        string string_signFeat = "signFeat-" + intToString(actual_signFeat);
        
        for(int test10 = 0; test10<nomiVideo.size(); test10++)
        {
            string nomeVideo;
            nomeVideo = nomiVideo.at(test10);
        
            string pathRes = "./results/" + pipeline + "/" + nomeVideo + "/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
            if(fileExists(pathRes.c_str()))
            {
                cv::FileStorage fs(pathRes, cv::FileStorage::READ);
        
                vector<bool> vr_punteggio16;
                vector<bool> vr_featuresSignatureCandidates;
                vector<bool> vr_signatureEnabled;
                vector<bool> vr_grayEnabled;
                vector<int> vr_matching_threshold;
                vector<int> vr_exp_positives;
                vector<int> vr_nFalsePositives;
                vector<int> vr_nFalseNegatives;
                
                cv::FileNode fn = fs["tests"];
                for (cv::FileNodeIterator i = fn.begin(); i != fn.end(); ++i)
                {
                    vr_punteggio16.push_back(((int)(*i)["punteggio16"]));
                    vr_featuresSignatureCandidates.push_back(((int)(*i)["featuresSignCand"]));
                    vr_signatureEnabled.push_back(((int)(*i)["signatureEnabled"]));
                    vr_grayEnabled.push_back(((int)(*i)["grayEnabled"]));
                    vr_matching_threshold.push_back((*i)["matching"]);
                    vr_exp_positives.push_back((*i)["Expected positives"]);
                    vr_nFalsePositives.push_back((*i)["False Positives"]);
                    vr_nFalseNegatives.push_back((*i)["False Negatives"]);
                }
                
                fs.release();
                
                

                for(int res = 0; res<vr_punteggio16.size(); res++)
                {
                    
                    string string_punteggio16 = "punteggio16-" + boolToString(vr_punteggio16.at(res));
                    string string_featuresSignatureCandidates = "featuresSignatureCandidates-" + boolToString(vr_featuresSignatureCandidates.at(res));
                    string string_signatureEnabled = "signatureEnabled-" + boolToString(vr_signatureEnabled.at(res));
                    string string_grayEnabled = "grayEnabled-" + boolToString(vr_grayEnabled.at(res));
                    string string_matching = "matching-" + intToString(vr_matching_threshold.at(res));
                    
                    string parameters = string_use63 + "_" + string_threshold + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_punteggio16 + "_" + string_featuresSignatureCandidates + "_" + string_signatureEnabled + "_" + string_grayEnabled + "_" + string_matching;
                    
                    pair<string,string> actual_pair = make_pair(nomeVideo, parameters);
                    
                    int actual_exp_positives = vr_exp_positives.at(res);
                    int actual_fp = vr_nFalsePositives.at(res);
                    int actual_fn = vr_nFalseNegatives.at(res);
                    int actual_fpfn = actual_fp + actual_fn;
                    
                    map_expectedPositive[actual_pair] = map_expectedPositive[actual_pair] + actual_exp_positives;
                    map_falsePositive[actual_pair] = map_falsePositive[actual_pair] + actual_fp;
                    map_falseNegative[actual_pair] = map_falseNegative[actual_pair] + actual_fn;
                    map_falsePosNeg[actual_pair] = map_falsePosNeg[actual_pair] + actual_fpfn;
                    
                }

            }
        }
    }
    cout<<endl;
    }  
    }  
    }
    }
    
    //////////////////////////////////////////////
    for(int iVideo = 0; iVideo<nomiVideo.size(); iVideo++)
    {
        string nomeVideo = nomiVideo.at(iVideo);
        
        //cerco i migliori falsi positivi del video corrente
        map<pair<string, string>, int>::iterator it = map_falsePositive.begin();
        int maxInt = std::numeric_limits<int>::max(); 
        int first_fp_best = maxInt;
        int second_fp_best = maxInt;
        int third_fp_best = maxInt;
        string first_string_fp_best = "";
        string second_string_fp_best = "";
        string third_string_fp_best = "";
        for(; it != map_falsePositive.end(); ++it)
        {
            pair<string, string> actual_pair = (*it).first;
            int actual_falsePositives = (*it).second;
            string actual_video = actual_pair.first;
            string actual_params = actual_pair.second;
            
            if(actual_video == nomeVideo)
            {
                if(actual_falsePositives < first_fp_best)
                {
                    third_fp_best = second_fp_best;
                    second_fp_best = first_fp_best;
                    first_fp_best = actual_falsePositives;
                    
                    third_string_fp_best = second_string_fp_best;
                    second_string_fp_best = first_string_fp_best;
                    first_string_fp_best = actual_params;
                }
                else
                {
                    if(actual_falsePositives < second_fp_best)
                    {
                        third_fp_best = second_fp_best;
                        second_fp_best = actual_falsePositives;
                        
                        third_string_fp_best = second_string_fp_best;
                        second_string_fp_best = actual_params;
                    }
                    else
                        if(actual_falsePositives < third_fp_best)
                        {
                            third_fp_best = actual_falsePositives;
                            third_string_fp_best = actual_params;
                        }
                }
            }
            
        }
        
        //cerco i migliori falsi negativi del video corrente
        it = map_falseNegative.begin();
        int first_fn_best = maxInt;
        int second_fn_best = maxInt;
        int third_fn_best = maxInt;
        string first_string_fn_best = "";
        string second_string_fn_best = "";
        string third_string_fn_best = "";
        for(; it != map_falseNegative.end(); ++it)
        {
            pair<string, string> actual_pair = (*it).first;
            int actual_falseNegatives = (*it).second;
            string actual_video = actual_pair.first;
            string actual_params = actual_pair.second;
            
            if(actual_video == nomeVideo)
            {
                if(actual_falseNegatives < first_fn_best)
                {
                    third_fn_best = second_fn_best;
                    second_fn_best = first_fn_best;
                    first_fn_best = actual_falseNegatives;
                    
                    third_string_fn_best = second_string_fn_best;
                    second_string_fn_best = first_string_fn_best;
                    first_string_fn_best = actual_params;
                }
                else
                {
                    if(actual_falseNegatives < second_fn_best)
                    {
                        third_fn_best = second_fn_best;
                        second_fn_best = actual_falseNegatives;
                        
                        third_string_fn_best = second_string_fn_best;
                        second_string_fn_best = actual_params;
                    }
                    else
                        if(actual_falseNegatives < third_fn_best)
                        {
                            third_fn_best = actual_falseNegatives;
                            third_string_fn_best = actual_params;
                        }
                }
            }
            
        }
        
        //cerco i migliori falsi positivi + falsi negativi del video corrente
        it = map_falsePosNeg.begin();
        int first_fpfn_best = maxInt;
        int second_fpfn_best = maxInt;
        int third_fpfn_best = maxInt;
        string first_string_fpfn_best = "";
        string second_string_fpfn_best = "";
        string third_string_fpfn_best = "";
        for(; it != map_falsePosNeg.end(); ++it)
        {
            pair<string, string> actual_pair = (*it).first;
            int actual_falsePosNegs = (*it).second;
            string actual_video = actual_pair.first;
            string actual_params = actual_pair.second;
            
            if(actual_video == nomeVideo)
            {
                if(actual_falsePosNegs < first_fpfn_best)
                {
                    third_fpfn_best = second_fpfn_best;
                    second_fpfn_best = first_fpfn_best;
                    first_fpfn_best = actual_falsePosNegs;
                    
                    third_string_fpfn_best = second_string_fpfn_best;
                    second_string_fpfn_best = first_string_fpfn_best;
                    first_string_fpfn_best = actual_params;
                }
                else
                {
                    if(actual_falsePosNegs < second_fpfn_best)
                    {
                        third_fpfn_best = second_fpfn_best;
                        second_fpfn_best = actual_falsePosNegs;
                        
                        third_string_fpfn_best = second_string_fpfn_best;
                        second_string_fpfn_best = actual_params;
                    }
                    else
                        if(actual_falsePosNegs < third_fpfn_best)
                        {
                            third_fpfn_best = actual_falsePosNegs;
                            third_string_fpfn_best = actual_params;
                        }
                }
            }
            
        }
        /*controllo expected
        map<pair<string, string>, int>::iterator itp = map_expectedPositive.begin();
        for(; itp != map_expectedPositive.end(); ++itp)
        {
            if(((*itp).first).first == nomeVideo)
            cout<<" - "<<(*itp).second<<" - "<<flush;
        }
        cout<<endl;*/
            
        string dirVideo = "./results/" + pipeline + "/" + nomeVideo + "/" + "BEST_RESULTS.yml";
        
        cv::FileStorage fs_best(dirVideo, cv::FileStorage::WRITE);
        fs_best << "BEST FALSE POSITIVES" << "[";
        
            fs_best << "{";
                fs_best << "position" << 1;
                fs_best << "parameters" << first_string_fp_best;
                fs_best << "results" << "-----";
                pair<string,string> first_fp_pair = make_pair(nomeVideo, first_string_fp_best);
                fs_best <<"Expected positives"<<map_expectedPositive[first_fp_pair];
                fs_best <<"False Positives"<<map_falsePositive[first_fp_pair];
                fs_best <<"False Negatives"<<map_falseNegative[first_fp_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[first_fp_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 2;
                fs_best << "parameters" << second_string_fp_best;
                fs_best << "results" << "-----";
                pair<string,string> second_fp_pair =  make_pair(nomeVideo, second_string_fp_best);
                fs_best <<"Expected positives"<<map_expectedPositive[second_fp_pair];
                fs_best <<"False Positives"<<map_falsePositive[second_fp_pair];
                fs_best <<"False Negatives"<<map_falseNegative[second_fp_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[second_fp_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 3;
                fs_best << "parameters" << third_string_fp_best;
                fs_best << "results" << "-----";
                pair<string,string> third_fp_pair =  make_pair(nomeVideo, third_string_fp_best);
                fs_best <<"Expected positives"<<map_expectedPositive[third_fp_pair];
                fs_best <<"False Positives"<<map_falsePositive[third_fp_pair];
                fs_best <<"False Negatives"<<map_falseNegative[third_fp_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[third_fp_pair];
            fs_best << "}";
        fs_best << "]";
        fs_best << "BEST FALSE NEGATIVES" << "[";
        
            fs_best << "{";
                fs_best << "position" << 1;
                fs_best << "parameters" << first_string_fn_best;
                fs_best << "results" << "-----";
                pair<string,string> first_fn_pair =  make_pair(nomeVideo, first_string_fn_best);
                fs_best <<"Expected positives"<<map_expectedPositive[first_fn_pair];
                fs_best <<"False Positives"<<map_falsePositive[first_fn_pair];
                fs_best <<"False Negatives"<<map_falseNegative[first_fn_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[first_fn_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 2;
                fs_best << "parameters" << second_string_fn_best;
                fs_best << "results" << "-----";
                pair<string,string> second_fn_pair =  make_pair(nomeVideo, second_string_fn_best);
                fs_best <<"Expected positives"<<map_expectedPositive[second_fn_pair];
                fs_best <<"False Positives"<<map_falsePositive[second_fn_pair];
                fs_best <<"False Negatives"<<map_falseNegative[second_fn_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[second_fn_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 3;
                fs_best << "parameters" << third_string_fn_best;
                fs_best << "results" << "-----";
                pair<string,string> third_fn_pair =  make_pair(nomeVideo, third_string_fn_best);
                fs_best <<"Expected positives"<<map_expectedPositive[third_fn_pair];
                fs_best <<"False Positives"<<map_falsePositive[third_fn_pair];
                fs_best <<"False Negatives"<<map_falseNegative[third_fn_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[third_fn_pair];
            fs_best << "}";
        fs_best << "]";
        fs_best << "BEST FALSE SUM" << "[";
            fs_best << "{";
                fs_best << "position" << 1;
                fs_best << "parameters" << first_string_fpfn_best;
                fs_best << "results" << "-----";
                pair<string,string> first_fpfn_pair =  make_pair(nomeVideo, first_string_fpfn_best);
                fs_best <<"Expected positives"<<map_expectedPositive[first_fpfn_pair];
                fs_best <<"False Positives"<<map_falsePositive[first_fpfn_pair];
                fs_best <<"False Negatives"<<map_falseNegative[first_fpfn_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[first_fpfn_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 2;
                fs_best << "parameters" << second_string_fpfn_best;
                fs_best << "results" << "-----";
                pair<string,string> second_fpfn_pair =  make_pair(nomeVideo, second_string_fpfn_best);
                fs_best <<"Expected positives"<<map_expectedPositive[second_fpfn_pair];
                fs_best <<"False Positives"<<map_falsePositive[second_fpfn_pair];
                fs_best <<"False Negatives"<<map_falseNegative[second_fpfn_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[second_fpfn_pair];
            fs_best << "}";
            fs_best << "{";
                fs_best << "position" << 3;
                fs_best << "parameters" << third_string_fpfn_best;
                fs_best << "results" << "-----";
                pair<string,string> third_fpfn_pair =  make_pair(nomeVideo, third_string_fpfn_best);
                fs_best <<"Expected positives"<<map_expectedPositive[third_fpfn_pair];
                fs_best <<"False Positives"<<map_falsePositive[third_fpfn_pair];
                fs_best <<"False Negatives"<<map_falseNegative[third_fpfn_pair];
                fs_best <<"FN_plus_FP"<<map_falsePosNeg[third_fpfn_pair];
            fs_best << "}";
        fs_best << "]";
        
    }
    
    
    
}

void analyzeResults_global(int nPipe, vector<string>& nomiVideo, vector<int> v_threshold_rgb, vector<bool> v_use63, vector<int> v_featuresUsed, vector<int> v_signFeat, vector<pair<string,string> > all_categories)
{
    string pipeline = "pipeline_" + intToString(nPipe);
    
    //< mappa<parametriUsati, valore_parametro>
    map<string, int> map_expectedPositive;
    map<string, int> map_falsePositive;
    map<string, int> map_falseNegative;
    map<string, int> map_falsePosNeg;    
    
    
    bool actual_use63;
    int actual_featuresUsed;
    
    int actual_signFeat;
    
    int actual_threshold_rgb;
    
    for(int del1 = 0; del1<v_threshold_rgb.size(); del1++)
	{
		actual_threshold_rgb = v_threshold_rgb.at(del1);
	    
	for(int del2 = 0; del2<v_use63.size(); del2++)
	{
		actual_use63 = v_use63.at(del2);
		
	for(int del3 = 0; del3<v_featuresUsed.size(); del3++)
	{
	    //cambio il numero di feature solo se use63 è false
	    if(actual_use63 == false)
            actual_featuresUsed = v_featuresUsed.at(del3);
	    else //use63 è true
            actual_featuresUsed = 63;
	    
	for(int del4 = 0; del4<v_signFeat.size(); del4++)
	{
		actual_signFeat = v_signFeat.at(del4);
    
    for(int w = 0; w<all_categories.size(); w++)
    {
        cout<<"."<<flush;
		pair<string,string> category = all_categories.at(w);
        string class_id = category.first + "_" + category.second;
        
        string string_use63 = "use63-" + boolToString(actual_use63);
        string string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
        string string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
        string string_signFeat = "signFeat-" + intToString(actual_signFeat);
        
        for(int test10 = 0; test10<nomiVideo.size(); test10++)
        {
            string nomeVideo;
            nomeVideo = nomiVideo.at(test10);
        
            string pathRes = "./results/" + pipeline + "/" + nomeVideo + "/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
            if(fileExists(pathRes.c_str()))
            {
                cv::FileStorage fs(pathRes, cv::FileStorage::READ);
        
                vector<bool> vr_punteggio16;
                vector<bool> vr_featuresSignatureCandidates;
                vector<bool> vr_signatureEnabled;
                vector<bool> vr_grayEnabled;
                vector<int> vr_matching_threshold;
                vector<int> vr_exp_positives;
                vector<int> vr_nFalsePositives;
                vector<int> vr_nFalseNegatives;
                
                cv::FileNode fn = fs["tests"];
                for (cv::FileNodeIterator i = fn.begin(); i != fn.end(); ++i)
                {
                    vr_punteggio16.push_back(((int)(*i)["punteggio16"]));
                    vr_featuresSignatureCandidates.push_back(((int)(*i)["featuresSignCand"]));
                    vr_signatureEnabled.push_back(((int)(*i)["signatureEnabled"]));
                    vr_grayEnabled.push_back(((int)(*i)["grayEnabled"]));
                    vr_matching_threshold.push_back((*i)["matching"]);
                    vr_exp_positives.push_back((*i)["Expected positives"]);
                    vr_nFalsePositives.push_back((*i)["False Positives"]);
                    vr_nFalseNegatives.push_back((*i)["False Negatives"]);
                }
                
                fs.release();
                
                

                for(int res = 0; res<vr_punteggio16.size(); res++)
                {
                    
                    string string_punteggio16 = "punteggio16-" + boolToString(vr_punteggio16.at(res));
                    string string_featuresSignatureCandidates = "featuresSignatureCandidates-" + boolToString(vr_featuresSignatureCandidates.at(res));
                    string string_signatureEnabled = "signatureEnabled-" + boolToString(vr_signatureEnabled.at(res));
                    string string_grayEnabled = "grayEnabled-" + boolToString(vr_grayEnabled.at(res));
                    string string_matching = "matching-" + intToString(vr_matching_threshold.at(res));
                    
                    string parameters = string_use63 + "_" + string_threshold + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_punteggio16 + "_" + string_featuresSignatureCandidates + "_" + string_signatureEnabled + "_" + string_grayEnabled + "_" + string_matching;
                    
                    int actual_exp_positives = vr_exp_positives.at(res);
                    int actual_fp = vr_nFalsePositives.at(res);
                    int actual_fn = vr_nFalseNegatives.at(res);
                    int actual_fpfn = actual_fp + actual_fn;
                    
                    map_expectedPositive[parameters] = map_expectedPositive[parameters] + actual_exp_positives;
                    map_falsePositive[parameters] = map_falsePositive[parameters] + actual_fp;
                    map_falseNegative[parameters] = map_falseNegative[parameters] + actual_fn;
                    map_falsePosNeg[parameters] = map_falsePosNeg[parameters] + actual_fpfn;
                    
                }

            }
        }
    }
    cout<<endl;
    }  
    }  
    }
    }
    
    //////////////////////////////////////////////
    
        
    //cerco i migliori falsi positivi
    map<string, int>::iterator it = map_falsePositive.begin();
    int maxInt = std::numeric_limits<int>::max(); 
    int first_fp_best = maxInt;
    int second_fp_best = maxInt;
    int third_fp_best = maxInt;
    string first_string_fp_best = "";
    string second_string_fp_best = "";
    string third_string_fp_best = "";
    for(; it != map_falsePositive.end(); ++it)
    {
        string actual_params = (*it).first;
        int actual_falsePositives = (*it).second;

        if(actual_falsePositives < first_fp_best)
        {
            third_fp_best = second_fp_best;
            second_fp_best = first_fp_best;
            first_fp_best = actual_falsePositives;
            
            third_string_fp_best = second_string_fp_best;
            second_string_fp_best = first_string_fp_best;
            first_string_fp_best = actual_params;
        }
        else
        {
            if(actual_falsePositives < second_fp_best)
            {
                third_fp_best = second_fp_best;
                second_fp_best = actual_falsePositives;
                
                third_string_fp_best = second_string_fp_best;
                second_string_fp_best = actual_params;
            }
            else
                if(actual_falsePositives < third_fp_best)
                {
                    third_fp_best = actual_falsePositives;
                    third_string_fp_best = actual_params;
                }
        }
        
    }
    
    //cerco i migliori falsi negativi
    it = map_falseNegative.begin();
    int first_fn_best = maxInt;
    int second_fn_best = maxInt;
    int third_fn_best = maxInt;
    string first_string_fn_best = "";
    string second_string_fn_best = "";
    string third_string_fn_best = "";
    for(; it != map_falseNegative.end(); ++it)
    {
        string actual_params = (*it).first;
        int actual_falseNegatives = (*it).second;

        if(actual_falseNegatives < first_fn_best)
        {
            third_fn_best = second_fn_best;
            second_fn_best = first_fn_best;
            first_fn_best = actual_falseNegatives;
            
            third_string_fn_best = second_string_fn_best;
            second_string_fn_best = first_string_fn_best;
            first_string_fn_best = actual_params;
        }
        else
        {
            if(actual_falseNegatives < second_fn_best)
            {
                third_fn_best = second_fn_best;
                second_fn_best = actual_falseNegatives;
                
                third_string_fn_best = second_string_fn_best;
                second_string_fn_best = actual_params;
            }
            else
                if(actual_falseNegatives < third_fn_best)
                {
                    third_fn_best = actual_falseNegatives;
                    third_string_fn_best = actual_params;
                }
        }
        
    }
    
    //cerco i migliori falsi positivi + falsi negativi
    it = map_falsePosNeg.begin();
    int first_fpfn_best = maxInt;
    int second_fpfn_best = maxInt;
    int third_fpfn_best = maxInt;
    string first_string_fpfn_best = "";
    string second_string_fpfn_best = "";
    string third_string_fpfn_best = "";
    for(; it != map_falsePosNeg.end(); ++it)
    {
        string actual_params = (*it).first;
        int actual_falsePosNegs = (*it).second;
        
        if(actual_falsePosNegs < first_fpfn_best)
        {
            third_fpfn_best = second_fpfn_best;
            second_fpfn_best = first_fpfn_best;
            first_fpfn_best = actual_falsePosNegs;
            
            third_string_fpfn_best = second_string_fpfn_best;
            second_string_fpfn_best = first_string_fpfn_best;
            first_string_fpfn_best = actual_params;
        }
        else
        {
            if(actual_falsePosNegs < second_fpfn_best)
            {
                third_fpfn_best = second_fpfn_best;
                second_fpfn_best = actual_falsePosNegs;
                
                third_string_fpfn_best = second_string_fpfn_best;
                second_string_fpfn_best = actual_params;
            }
            else
                if(actual_falsePosNegs < third_fpfn_best)
                {
                    third_fpfn_best = actual_falsePosNegs;
                    third_string_fpfn_best = actual_params;
                }
        }

    }
    
        
    string dirVideo = "./results/" + pipeline + "/BEST_RESULTS.yml";
    
    cv::FileStorage fs_best(dirVideo, cv::FileStorage::WRITE);
    fs_best << "BEST FALSE POSITIVES" << "[";
    
        fs_best << "{";
            fs_best << "position" << 1;
            fs_best << "parameters" << first_string_fp_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[first_string_fp_best];
            fs_best <<"False Positives"<<map_falsePositive[first_string_fp_best];
            fs_best <<"False Negatives"<<map_falseNegative[first_string_fp_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[first_string_fp_best];
        fs_best << "}";
        fs_best << "{";
            fs_best << "position" << 2;
            fs_best << "parameters" << second_string_fp_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[second_string_fp_best];
            fs_best <<"False Positives"<<map_falsePositive[second_string_fp_best];
            fs_best <<"False Negatives"<<map_falseNegative[second_string_fp_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[second_string_fp_best];
        fs_best << "}";
        fs_best << "{";
            fs_best << "position" << 3;
            fs_best << "parameters" << third_string_fp_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[third_string_fp_best];
            fs_best <<"False Positives"<<map_falsePositive[third_string_fp_best];
            fs_best <<"False Negatives"<<map_falseNegative[third_string_fp_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[third_string_fp_best];
        fs_best << "}";
    fs_best << "]";
    fs_best << "BEST FALSE NEGATIVES" << "[";
    
        fs_best << "{";
            fs_best << "position" << 1;
            fs_best << "parameters" << first_string_fn_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[first_string_fn_best];
            fs_best <<"False Positives"<<map_falsePositive[first_string_fn_best];
            fs_best <<"False Negatives"<<map_falseNegative[first_string_fn_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[first_string_fn_best];
        fs_best << "}";
        fs_best << "{";
            fs_best << "position" << 2;
            fs_best << "parameters" << second_string_fn_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[second_string_fn_best];
            fs_best <<"False Positives"<<map_falsePositive[second_string_fn_best];
            fs_best <<"False Negatives"<<map_falseNegative[second_string_fn_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[second_string_fn_best];
        fs_best << "}";
        fs_best << "{";
            fs_best << "position" << 3;
            fs_best << "parameters" << third_string_fn_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[third_string_fn_best];
            fs_best <<"False Positives"<<map_falsePositive[third_string_fn_best];
            fs_best <<"False Negatives"<<map_falseNegative[third_string_fn_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[third_string_fn_best];
        fs_best << "}";
    fs_best << "]";
    fs_best << "BEST FALSE SUM" << "[";
        fs_best << "{";
            fs_best << "position" << 1;
            fs_best << "parameters" << first_string_fpfn_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[first_string_fpfn_best];
            fs_best <<"False Positives"<<map_falsePositive[first_string_fpfn_best];
            fs_best <<"False Negatives"<<map_falseNegative[first_string_fpfn_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[first_string_fpfn_best];
        fs_best << "}";
        fs_best << "{";
            fs_best << "position" << 2;
            fs_best << "parameters" << second_string_fpfn_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[second_string_fpfn_best];
            fs_best <<"False Positives"<<map_falsePositive[second_string_fpfn_best];
            fs_best <<"False Negatives"<<map_falseNegative[second_string_fpfn_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[second_string_fpfn_best];
        fs_best << "}";
        fs_best << "{";
            fs_best << "position" << 3;
            fs_best << "parameters" << third_string_fpfn_best;
            fs_best << "results" << "-----";
            fs_best <<"Expected positives"<<map_expectedPositive[third_string_fpfn_best];
            fs_best <<"False Positives"<<map_falsePositive[third_string_fpfn_best];
            fs_best <<"False Negatives"<<map_falseNegative[third_string_fpfn_best];
            fs_best <<"FN_plus_FP"<<map_falsePosNeg[third_string_fpfn_best];
        fs_best << "}";
    fs_best << "]";
        
    
    
    
    
}

 void pulisci_risultati(int nPipe, bool& actual_use63, int& actual_featuresUsed, int& actual_signFeat, bool& actual_punteggio16, bool& actual_featuresSignatureCandidates, bool& actual_grayEnabled, bool& actual_signatureEnabled, int& actual_threshold_rgb, int& actual_matching_threshold, string class_id, string nomeVideo)
 {
    for(int nm = 0; nm < nPipe; nm++)
    {
        string string_use63 = "use63-" + boolToString(actual_use63);
        string string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
        string string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
        string string_signFeat = "signFeat-" + intToString(actual_signFeat);

        string pipeline = "pipeline_" + intToString(nm);
        string pathRes = "./results/" + pipeline + "/" + nomeVideo + "/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
        if(fileExists(pathRes.c_str()))
        {
            cv::FileStorage fs(pathRes, cv::FileStorage::READ);
                
            vector<bool> vr_punteggio16;
            vector<bool> vr_featuresSignatureCandidates;
            vector<bool> vr_signatureEnabled;
            vector<bool> vr_grayEnabled;
            vector<int> vr_matching_threshold;
            vector<int> vr_exp_positives;
            vector<int> vr_nFalsePositives;
            vector<int> vr_nFalseNegatives;
            
            vector<bool> vr_punteggio16_clean;
            vector<bool> vr_featuresSignatureCandidates_clean;
            vector<bool> vr_signatureEnabled_clean;
            vector<bool> vr_grayEnabled_clean;
            vector<int> vr_matching_threshold_clean;
            vector<int> vr_exp_positives_clean;
            vector<int> vr_nFalsePositives_clean;
            vector<int> vr_nFalseNegatives_clean;
            cout<<pathRes<<endl;
            cv::FileNode fn = fs["tests"];
            for (cv::FileNodeIterator i = fn.begin(); i != fn.end(); ++i)
            {
                vr_punteggio16.push_back(((int)(*i)["punteggio16"]));
                vr_featuresSignatureCandidates.push_back(((int)(*i)["featuresSignCand"]));
                vr_signatureEnabled.push_back(((int)(*i)["signatureEnabled"]));
                vr_grayEnabled.push_back(((int)(*i)["grayEnabled"]));
                vr_matching_threshold.push_back((*i)["matching"]);
                vr_exp_positives.push_back((*i)["Expected positives"]);
                vr_nFalsePositives.push_back((*i)["False Positives"]);
                vr_nFalseNegatives.push_back((*i)["False Negatives"]);
            }
            fs.release();
            
            
            
            for(int in = 0; in <vr_punteggio16.size(); in++)
            {
                bool isAlreadyInside = false;
                //controllo se il result con questi parametri non è già presente
                for(int in2 = in+1; in2 <vr_punteggio16.size(); in2++)
                {
                    if(vr_punteggio16.at(in) == vr_punteggio16.at(in2) &&
                        vr_featuresSignatureCandidates.at(in) == vr_featuresSignatureCandidates.at(in2) &&
                        vr_signatureEnabled.at(in) == vr_signatureEnabled.at(in2) &&
                        vr_grayEnabled.at(in) == vr_grayEnabled.at(in2) &&
                        vr_matching_threshold.at(in) == vr_matching_threshold.at(in2))
                    {
                        isAlreadyInside = true;
                    }
                }
                
                if(isAlreadyInside == false)
                {
                    vr_punteggio16_clean.push_back(vr_punteggio16.at(in));
                    vr_featuresSignatureCandidates_clean.push_back(vr_featuresSignatureCandidates.at(in));
                    vr_signatureEnabled_clean.push_back(vr_signatureEnabled.at(in));
                    vr_grayEnabled_clean.push_back(vr_grayEnabled.at(in));
                    vr_matching_threshold_clean.push_back(vr_matching_threshold.at(in));
                    vr_exp_positives_clean.push_back(vr_exp_positives.at(in));
                    vr_nFalsePositives_clean.push_back(vr_nFalsePositives.at(in));
                    vr_nFalseNegatives_clean.push_back(vr_nFalseNegatives.at(in));
                }
                
            }
            
            cv::FileStorage fs_new(pathRes, cv::FileStorage::WRITE);
            fs_new << "tests" << "[";
            for(int res = 0; res<vr_punteggio16_clean.size(); res++)
            {
            
            
                fs_new << "{";
                
                fs_new << "Test" << res;
                fs_new << "signatureEnabled" << vr_signatureEnabled_clean.at(res);
                fs_new << "grayEnabled" << vr_grayEnabled_clean.at(res);
                fs_new << "featuresSignCand" << vr_featuresSignatureCandidates_clean.at(res);
                fs_new << "matching"  << vr_matching_threshold_clean.at(res);
                fs_new << "punteggio16"  << vr_punteggio16_clean.at(res);
                fs_new << "results" << "-----";
                fs_new <<"Expected positives"<<vr_exp_positives_clean.at(res);
                fs_new <<"False Positives"<<vr_nFalsePositives_clean.at(res);
                fs_new <<"False Negatives"<<vr_nFalseNegatives_clean.at(res);
                
                fs_new << "}"; // current result
            
            
            }
            fs_new << "]"; // tests
            fs_new.release();
        
        }
    }
    
    
     
 }
void saveResults(bool& actual_use63, int& actual_featuresUsed, int& actual_signFeat, bool& actual_punteggio16, bool& actual_featuresSignatureCandidates, bool& actual_grayEnabled, bool& actual_signatureEnabled, int& actual_threshold_rgb, int& actual_matching_threshold, vector<VideoResult> videoResults, string class_id, string nomeVideo){
    
    for(int nm = 0; nm<videoResults.size(); nm++)
    {
        VideoResult videoResult = videoResults[nm];
        
        string string_use63 = "use63-" + boolToString(actual_use63);
        string string_threshold = "thresholdRGB-" + intToString(actual_threshold_rgb);
        string string_featuresUsed = "featuresUsed-" + intToString(actual_featuresUsed);
        string string_signFeat = "signFeat-" + intToString(actual_signFeat);

        string pipeline = "pipeline_" + intToString(nm);

        string pathRes = "./results/" + pipeline + "/" + nomeVideo + "/" + class_id + "/" + string_use63 + "_" + string_featuresUsed + "_" + string_signFeat + "_" + string_threshold + ".yml";
        cout<<pathRes<<endl;
        if(!fileExists(pathRes.c_str()))
        {
            cv::FileStorage fs(pathRes, cv::FileStorage::WRITE);
            fs << "tests" << "[";
            
            fs << "{";
            
            fs << "Test" << 0;
            fs << "signatureEnabled" << actual_signatureEnabled;
            fs << "grayEnabled" << actual_grayEnabled;
            fs << "featuresSignCand" << actual_featuresSignatureCandidates;
            fs << "matching"  << actual_matching_threshold;
            fs << "punteggio16"  << actual_punteggio16;
            fs << "results" << "------";
            fs <<"Expected positives"<<videoResult.positivesExpected;
            fs <<"False Positives"<<videoResult.nFalsePositive;
            fs <<"False Negatives"<<videoResult.nFalseNegative;
            
            fs << "}"; // current result
            
            fs << "]"; // tests
        }
        else
        {
            cv::FileStorage fs(pathRes, cv::FileStorage::READ);
            
            vector<bool> vr_punteggio16;
            vector<bool> vr_featuresSignatureCandidates;
            vector<bool> vr_signatureEnabled;
            vector<bool> vr_grayEnabled;
            vector<int> vr_matching_threshold;
            vector<int> vr_exp_positives;
            vector<int> vr_nFalsePositives;
            vector<int> vr_nFalseNegatives;
            
            cv::FileNode fn = fs["tests"];
            for (cv::FileNodeIterator i = fn.begin(); i != fn.end(); ++i)
            {
                vr_punteggio16.push_back(((int)(*i)["punteggio16"]));
                vr_featuresSignatureCandidates.push_back(((int)(*i)["featuresSignCand"]));
                vr_signatureEnabled.push_back(((int)(*i)["signatureEnabled"]));
                vr_grayEnabled.push_back(((int)(*i)["grayEnabled"]));
                vr_matching_threshold.push_back((*i)["matching"]);
                vr_exp_positives.push_back((*i)["Expected positives"]);
                vr_nFalsePositives.push_back((*i)["False Positives"]);
                vr_nFalseNegatives.push_back((*i)["False Negatives"]);
            }
            
            fs.release();
            
            bool isAlreadyInside = false;
            //controllo se il result con questi parametri non è già presente
            
            for(int in = 0; in <vr_punteggio16.size(); in++)
            {
                if(actual_punteggio16 == vr_punteggio16.at(in) &&
                    actual_featuresSignatureCandidates == vr_featuresSignatureCandidates.at(in) &&
                    actual_signatureEnabled == vr_signatureEnabled.at(in) &&
                    actual_grayEnabled == vr_grayEnabled.at(in) &&
                    actual_matching_threshold == vr_matching_threshold.at(in))
                {
                    isAlreadyInside = true;
                    break;
                }
            }
            
            if(isAlreadyInside == false)
            {
                vr_punteggio16.push_back(actual_punteggio16);
                vr_featuresSignatureCandidates.push_back(actual_featuresSignatureCandidates);
                vr_signatureEnabled.push_back(actual_signatureEnabled);
                vr_grayEnabled.push_back(actual_grayEnabled);
                vr_matching_threshold.push_back(actual_matching_threshold);
                vr_exp_positives.push_back(videoResult.positivesExpected);
                vr_nFalsePositives.push_back(videoResult.nFalsePositive);
                vr_nFalseNegatives.push_back(videoResult.nFalseNegative);
            }
            
            cv::FileStorage fs_new(pathRes, cv::FileStorage::WRITE);
            fs_new << "tests" << "[";
            for(int res = 0; res<vr_punteggio16.size(); res++)
            {
            
            
                fs_new << "{";
                
                fs_new << "Test" << res;
                fs_new << "signatureEnabled" << vr_signatureEnabled.at(res);
                fs_new << "grayEnabled" << vr_grayEnabled.at(res);
                fs_new << "featuresSignCand" << vr_featuresSignatureCandidates.at(res);
                fs_new << "matching"  << vr_matching_threshold.at(res);
                fs_new << "punteggio16"  << vr_punteggio16.at(res);
                fs_new << "results" << "-----";
                fs_new <<"Expected positives"<<vr_exp_positives.at(res);
                fs_new <<"False Positives"<<vr_nFalsePositives.at(res);
                fs_new <<"False Negatives"<<vr_nFalseNegatives.at(res);
                
                fs_new << "}"; // current result
            
            
            }
            fs_new << "]"; // tests
            fs_new.release();
        }
    }
}




bool contains(vector<int> vc, int n)
{
    for(int i = 0; i<vc.size(); i++)
        if(vc[i] == n)
            return true;
    
    return false;
}

void analyzeResults_pipelines(int nPipe)
{
    int maxInt = std::numeric_limits<int>::max(); 
    int best_fp[nPipe];
    int best_fn[nPipe];
    int best_fpfn[nPipe];
    for(int nm = 0; nm<nPipe; nm++)
    {
        string pipeline = "pipeline_" + intToString(nm);
        string pathRes = "./results/" + pipeline + "/BEST_RESULTS.yml";
        if(fileExists(pathRes.c_str()))
        {
            cv::FileStorage fs(pathRes, cv::FileStorage::READ);
            
            cv::FileNode fn = fs["BEST FALSE POSITIVES"];
            cv::FileNodeIterator i = fn.begin();
            if(((int)(*i)["position"]) == 1)
                best_fp[nm] = (int)(*i)["False Positives"];
            
            fn = fs["BEST FALSE NEGATIVES"];
            i = fn.begin();
            if(((int)(*i)["position"]) == 1)
                best_fn[nm] = (int)(*i)["False Negatives"];
            
            fn = fs["BEST FALSE SUM"];
            i = fn.begin();
            if(((int)(*i)["position"]) == 1)
                best_fpfn[nm] = (int)(*i)["FN_plus_FP"];
            
            fs.release();
        }
        
            else
                cout<<"Pipeline does not exist"<<endl;
    }
    
    vector<int> bestIndexFP;
    vector<int> bestIndexFN;
    vector<int> bestIndexFPFN;
    for(int i = 0; i<nPipe; i++)
    {
        int tmpBestFP = maxInt;
        int tmpBestFN = maxInt;
        int tmpBestFPFN = maxInt;
        
        int nextFP;
        int nextFN;
        int nextFPFN;
        for(int j = 0; j<nPipe; j++)
        {
            if(best_fp[j] < tmpBestFP && contains(bestIndexFP, j) == false)
            {
                tmpBestFP = best_fp[j];
                nextFP = j;
            }
            if(best_fn[j] < tmpBestFN && contains(bestIndexFN, j) == false)
            {
                tmpBestFN = best_fn[j];
                nextFN = j;
            }
            if(best_fpfn[j] < tmpBestFPFN && contains(bestIndexFPFN, j) == false)
            {
                tmpBestFPFN = best_fpfn[j];
                nextFPFN = j;
            }
        }
        
        bestIndexFP.push_back(nextFP);
        bestIndexFN.push_back(nextFN);
        bestIndexFPFN.push_back(nextFPFN);
        
    }
    
    
    string dirMain = "./results/BEST_PIPELINES.yml";
    
    cv::FileStorage fs_best(dirMain, cv::FileStorage::WRITE);
    fs_best << "BEST FALSE POSITIVES" << "[";
    int position = 1;
    for(int nm = 0; nm<nPipe; nm++)
    {
        fs_best << "{";
            fs_best << "position" << position;
            fs_best << "PIPELINE" << bestIndexFP[nm];
            fs_best <<"False Positives"<<best_fp[bestIndexFP[nm]];
        fs_best << "}";
        position++;
    }
    fs_best << "]";
    
    fs_best << "BEST FALSE NEGATIVES" << "[";
        position = 1;
        for(int nm = 0; nm<nPipe; nm++)
        {
            fs_best << "{";
                fs_best << "position" << position;
                fs_best << "PIPELINE" << bestIndexFN[nm];
                fs_best <<"False Negatives"<<best_fn[bestIndexFN[nm]];
            fs_best << "}";
            position++;
        }
    fs_best << "]";
    fs_best << "BEST FALSE SUM" << "[";
        position = 1;
        for(int nm = 0; nm<nPipe; nm++)
        {
            fs_best << "{";
                fs_best << "position" << position;
                fs_best << "PIPELINE" << bestIndexFPFN[nm];
                fs_best <<"FN_plus_FP"<<best_fpfn[bestIndexFPFN[nm]];
            fs_best << "}";
            position++;
        }
    fs_best << "]";
    
    fs_best.release();
        
    
    
    
    
}
