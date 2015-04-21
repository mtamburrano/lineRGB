#include <iterator>
#include <set>
#include <cstdio>
#include <sstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/rgbd/linemod.hpp>
#include "objdetect_line_rgb.hpp"

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

static const bool DEBUG = false;

static void help_f() {
  printf(
      "\n\n"
      " Usage: line_test [options] [actions] [modality] \n\n"
      " This test can run a Line2D or a LineRGB detector on a sequence of images to detect,\n"
      " a trained objects in the scene. The templates of the object can be provided using the\n"
      " --train option or loading a .yml file by the --load option. The .yml file can be generated \n"
      " using together the --train and --save options, or pressing 'w' during test execution; in \n"
      " both cases the file will be stored in the same folder of this executable. \n\n"
      " usage example 1: line_test -t [templates_folder] -v [video_folder] --save --train --test --linergb\n"
      " usage example 2: line_test -v [video_folder] --load --test --linergb\n"
      " usage example 3: line_test -t [templates_folder] --save --line2d\n\n"
      "Options:\n"
      "\t -h  	   -- This help page\n"
      "\t -t  	   -- provides the path to the templates folder\n"
      "\t -v  	   -- provides the path to the video folder\n"
      "\t --train   -- trains a detector with the chosen modality using the templates folder specified by -t\n"
      "\t --test    -- search for an object trained in a video specified by -v, using the chosen modality\n"
      "\t --load    -- load a detector of the chosen modality using a pre-saved .yml\n"
      "\t --save    -- write a detector of the chosen modality using the templates just trained\n"
      "\t --linergb -- specifies to use the LineRGB modality\n"
      "\t --line2d  -- specifies to use the original Line2D modality\n\n"
      "\t --linemodrgb -- specifies to use the LinemodRGB modality\n\n"
      "\t --check_groups -- display a summary of template groups\n\n"
      "Keys on test running:\n"
      "\t w   -- write the detector on a file\n"
      "\t p   -- pause the test\n"
      "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities
class Timer {
 public:
  Timer()
      : start_(0),
        time_(0) {
  }

  void start() {
    start_ = cv::getTickCount();
  }

  void stop() {
    CV_Assert(start_ != 0);
    int64 end = cv::getTickCount();
    time_ += end - start_;
    start_ = 0;
  }

  double time() {
    double ret = time_ / cv::getTickFrequency();
    time_ = 0;
    return ret;
  }

 private:
  int64 start_, time_;
};

int stringToInt(string s) {
  stringstream ss;
  int number;
  ss << s;
  ss >> number;

  return number;
}

string i2s(int number) {
  stringstream ss;
  string s;
  ss << number;
  ss >> s;

  return s;
}

string charToString(char* c) {
  stringstream ss;
  string s;
  ss << c;
  ss >> s;

  return s;
}

std::string fixedLength(int value, int digits) {
  unsigned int uvalue = value;
  if (value < 0) {
    uvalue = -uvalue;
  }
  std::string result;
  while (digits-- > 0) {
    result += ('0' + uvalue % 10);
    uvalue /= 10;
  }
  if (value < 0) {
    result += '-';
  }
  std::reverse(result.begin(), result.end());
  return result;
}

void drawResponseLineRGB(const std::vector<cv::line_rgb::Template>& templates,
                         int num_modalities, cv::Mat& dst, cv::Point offset,
                         int T, short rejected, string class_id,
                         bool only_non_border) {

  cv::Scalar colorT;
  for (int m = 0; m < num_modalities; ++m) {

    vector<cv::line_rgb::Feature> all_features;
    all_features.insert(all_features.end(),
                        templates[m].features_border.begin(),
                        templates[m].features_border.end());
    all_features.insert(all_features.end(),
                        templates[m].features_inside.begin(),
                        templates[m].features_inside.end());

    int n_total_features = all_features.size()
        + templates[m].color_features.size();
    vector<cv::line_rgb::Feature> features;
    features.insert(features.end(), all_features.begin(), all_features.end());
    features.insert(features.end(), templates[m].color_features.begin(),
                    templates[m].color_features.end());

    for (int i = 0; i < (int) n_total_features; ++i) {
      cv::line_rgb::Feature f = features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      switch (f.rgb_label) {
        case 0:
          colorT = CV_RGB(255, 0, 0);
          break;
        case 1:
          colorT = CV_RGB(0, 255, 0);
          break;
        case 2:
          colorT = CV_RGB(0, 0, 255);
          break;
        case 3:
          colorT = CV_RGB(255, 255, 0);
          break;
        case 4:
          colorT = CV_RGB(255, 0, 255);
          break;
        case 5:
          colorT = CV_RGB(0, 255, 255);
          break;
        case 6:
          colorT = CV_RGB(255, 255, 255);
          break;
        case 7:
          colorT = CV_RGB(0, 0, 0);
          break;
      }
      if (m == 0) {
        if (f.on_border == false || only_non_border == false)
          cv::circle(dst, pt, T / 2, colorT);
        else
          cv::rectangle(dst, Point(pt.x - 1, pt.y - 1),
                        cv::Point(pt.x + 1, pt.y + 1), CV_RGB(255, 255, 255));
      }
      if (m == 1)
        cv::rectangle(dst, Point(pt.x, pt.y), cv::Point(pt.x + 1, pt.y + 1),
                      CV_RGB(127, 127, 127));
    }

  }
}

void drawResponseLine2D(const std::vector<cv::linemod::Template>& templates,
                        int num_modalities, cv::Mat& dst, cv::Point offset,
                        int T, short rejected, string class_id) {

  cv::Scalar color;
  for (int m = 0; m < num_modalities; ++m) {

    for (int i = 0; i < (int) templates[m].features.size(); ++i) {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);

      color = CV_RGB(0, 0, 255);

      cv::circle(dst, pt, T / 2, color);
    }

  }
}

string last_occur(string str1, const char* char_split) {
  unsigned found = str1.find_last_of(char_split);

  return str1.substr(found + 1);
}

// Functions to store detector and templates in single XML/YAML file
cv::Ptr<cv::line_rgb::Detector> readLineRGB(const std::string& filename) {
  cv::Ptr<cv::line_rgb::Detector> detector =
      cv::makePtr<cv::line_rgb::Detector>();
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector->readClass(*i);

  return detector;
}

void writeLineRGB(const cv::Ptr<cv::line_rgb::Detector>& detector,
                  const std::string& filename, bool use_hsv) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs, use_hsv);

  cout <<"SSSSSSSSSSSALVO" << endl;

  std::vector<cv::String> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int) ids.size(); ++i) {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}";  // current class
  }
  fs << "]";  // classes
  fs.releaseAndGetString();
}

// Functions to store detector and templates in single XML/YAML file
static cv::Ptr<cv::linemod::Detector> readLine2D(const std::string& filename) {
  cv::Ptr<cv::linemod::Detector> detector =
      cv::makePtr<cv::linemod::Detector>();
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector->readClass(*i);

  return detector;
}

static void writeLine2D(const cv::Ptr<cv::linemod::Detector>& detector,
                        const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs);

  std::vector<cv::String> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int) ids.size(); ++i) {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}";  // current class
  }
  fs << "]";  // classes
  fs.releaseAndGetString();
}

Mat rotateImage(const Mat& source, double angle) {
  Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
  Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
  Mat dst;
  warpAffine(source, dst, rot_mat, source.size(), INTER_CUBIC);
  return dst;
}

void readDirectory(string dirpath, vector<string>& list_files,
                   bool searchDirectories) {

  for (boost::filesystem::directory_iterator it(dirpath);
      it != boost::filesystem::directory_iterator(); ++it) {
    if (searchDirectories == false
        && boost::filesystem::is_regular_file(it->status())) {
      list_files.push_back((string) it->path().c_str());
    } else if (searchDirectories == true
        && boost::filesystem::is_directory(it->status())) {
      list_files.push_back((string) it->path().c_str());
    }
  }

}

TemplatesObject* getImagePathsFromListFile(vector<string>& list_files,
                                           string name, FileStorage fs_pose,
                                           bool with_depth) {

  TemplatesObject* t = new TemplatesObject();
  for (int i = 0; i < list_files.size(); i++) {
    string temp_string = list_files.at(i);
    if (temp_string.find("mask") != string::npos) {
      t->masks.push_back(temp_string);
    }
    if (temp_string.find("image") != string::npos) {
      t->images.push_back(temp_string);
    }
    if (with_depth) {
      if (temp_string.find("depth") != string::npos) {
        //search also for yml, because depth.png are only for debug purposes
        if (temp_string.find("yml") != string::npos) {
          t->depths.push_back(temp_string);
        }
      }
    }
  }

  if (t->masks.size() != t->images.size()) {
    cout << "mask and imagesmust have the same size" << endl;
    CV_Assert(false);
  }
  if (with_depth) {
    if (t->depths.size() != t->images.size()) {
      cout << "images and depths must have the same size" << endl;
      CV_Assert(false);
    }
  }

  t->name = name;

  FileNode node_dataset = fs_pose["dataset"];
  int id = 0;
  for (FileNodeIterator it = node_dataset.begin(); it != node_dataset.end();
      ++it, id++) {
    FileNode nextPose = (*it);
    Pose* p = new Pose();
    nextPose["radius"] >> p->radius;
    nextPose["Tx"] >> p->Tx;
    nextPose["Ty"] >> p->Ty;
    nextPose["Tz"] >> p->Tz;
    nextPose["angle"] >> p->angle;
    t->poses.push_back(*p);
    delete (p);
  }
  fs_pose.release();

  if (t->poses.size() != t->images.size()) {
    cout << "poses and images must have the same size" << endl;
    CV_Assert(false);
  }

  /*for (int i = 0; i < t->poses.size(); i++) {
    cout << "pose" << i << ": " << endl;
    cout << "     radius: " << t->poses[i].radius << endl;
    cout << "     Tx: " << t->poses[i].Tx << endl;
    cout << "     Ty: " << t->poses[i].Ty << endl;
    cout << "     Tz: " << t->poses[i].Tz << endl;
    cout << "     angle: " << t->poses[i].angle << endl;
  }*/


  sort(t->masks.begin(), t->masks.end());
  sort(t->images.begin(), t->images.end());
  if (with_depth) {
    sort(t->depths.begin(), t->depths.end());
  }

  return t;
}

void getAllObjectsTemplates(string templates_folder,
                            vector<TemplatesObject*>& all_objects_templates,
                            bool with_depth) {
  vector<string> list_dirs;
  //with true readDirectory searchs for directories
  readDirectory(templates_folder, list_dirs, true);
  vector<string> list_files;
  //each folder corresponds to an object
  for (int i = 0; i < list_dirs.size(); i++) {
    string folder = list_dirs[i];
    cout << "folder: " << folder << endl;
    //with false readDirectory searchs for regular files
    readDirectory(folder.c_str(), list_files, false);
    string folder_name = last_occur(folder, "/");
    FileStorage fs_pose(folder + "/output.xml", cv::FileStorage::READ);
    all_objects_templates.push_back(
        getImagePathsFromListFile(list_files, folder_name, fs_pose,
                                  with_depth));
  }

}

void splitByUnderscore(string src, string& word1, string& word2) {
  string tmp = last_occur(src, "_");

  word1 = src.substr(0, src.size() - tmp.size());
  word2 = tmp.substr(1, 1);
}

void splitBySlash(string src, string& word1, string& word2) {
  string tmp = last_occur(src, "/");
  if (tmp == "/") {
    src = src.substr(0, src.size() - 1);
    tmp = last_occur(src, "/");
    src = src + "/";
  } else {
    src = src + "/";
  }

  word1 = src;
  word2 = tmp.substr(1, tmp.size() - 1);
}

bool sortbygid(pair<cv::line_rgb::Template, int> t1,
               pair<cv::line_rgb::Template, int> t2) {
  return (t1.first.id_group < t2.first.id_group);
}

void do_check_groups(bool isLineRGB, string template_folder) {
  string filename;
  if (isLineRGB) {
    filename = "line_rgb_templates.yml";
  } else {
    filename = "linemod_rgb_templates.yml";
  }

  cv::Ptr<cv::line_rgb::Detector> detector_linergb = readLineRGB(filename);
  std::vector<cv::String> ids = detector_linergb->classIds();
  int num_classes = detector_linergb->numClasses();
  cout << "Loaded " << filename << " with " << num_classes << " classes and "
       << detector_linergb->numTemplates() << endl;
  int num_modalities = detector_linergb->getModalities().size();
  string pathof_templ_paths = "PATHS_" + filename;
  FileStorage templ_paths(pathof_templ_paths, cv::FileStorage::READ);
  cout << "reading " << pathof_templ_paths << endl;

  for (int i = 0; i < ids.size(); i++) {
    cv::String class_id = ids[i];
    int num_templates = detector_linergb->numTemplates(class_id);
    vector<pair<cv::line_rgb::Template, int> > sorted_bygid_templates;

    for (int template_id = 0; template_id < num_templates; template_id++) {
      std::vector<cv::line_rgb::Template> templates = detector_linergb
          ->getTemplates(class_id, template_id);
      sorted_bygid_templates.push_back(make_pair(templates[0], template_id));

    }

    sort(sorted_bygid_templates.begin(), sorted_bygid_templates.end(),
         sortbygid);
    int id_group = -1;
    int group_count = 0;
    int num_groups = 0;
    vector<pair<string, int> > group_paths;
    for (int nt = 0; nt < sorted_bygid_templates.size(); nt++) {

      pair<cv::line_rgb::Template, int> t = sorted_bygid_templates[nt];
      int template_id = t.second;
      cv::String index_template_path = class_id + "_" + i2s(template_id);
      string template_path;
      templ_paths[index_template_path] >> template_path;

      if (id_group != t.first.id_group) {
        if (id_group != -1) {
          cout << "ID GROUP: " << id_group << endl;
          cout << "       " << group_count << " elements" << endl;
          for (int gp = 0; gp < group_paths.size(); gp++) {
            cout << "             " << group_paths[gp].first << " - t_id: "
                 << group_paths[gp].second << endl;
          }

        }
        id_group = t.first.id_group;
        group_count = 1;
        num_groups++;
        group_paths.clear();
        group_paths.push_back(make_pair(template_path, template_id));
      } else {
        group_count++;
        group_paths.push_back(make_pair(template_path, template_id));
      }

      /*int template_id = t.second;
       cv::String index_template_path = class_id+"_"+i2s(template_id);
       string template_path;
       templ_paths[index_template_path] >> template_path;

       cout << index_template_path <<" template_path: "<<template_path<<endl;
       Mat rgb = imread(template_path, 1);

       //vector<cv::line_rgb::Template> template_to_draw;
       //template_to_draw.push_back(t.first);
       //drawResponseLineRGB(template_to_draw, num_modalities, rgb,
       //		cv::Point(0, 0), 8, -1, class_id,
       //		false);

       cv::String name_window = class_id+"-GID_"+i2s(t.first.id_group);
       imshow(name_window, rgb);
       waitKey();*/

    }
    cout << "NUM GROUPS: " << num_groups << endl;

  }
}

//////////////////MAIN///////////////////////

int main(int argc, char * argv[]) {
  if (argc <= 2) {
    help_f();
    return 0;
  }

  bool line2d = false;
  bool linemod = false;
  bool linergb = false;
  bool linemodrgb = false;
  bool hsv = false;
  bool rgb = false;
  bool only_non_border = false;
  bool check_groups = false;
  bool help = false;

  bool GROUP_SIMILAR_TEMPLATES = true;

  string path_name = "";
  string path_templates = "";

  for (int h = 1; h <= (argc - 1); h++) {
    if (strcmp("-t", argv[h]) == 0) {
      path_templates = charToString(argv[h + 1]);
      h++;
    }

    if (strcmp("--line2d", argv[h]) == 0) {
      line2d = true;
    }
    if (strcmp("--linergb", argv[h]) == 0) {
      linergb = true;
    }
    if (strcmp("--linemodrgb", argv[h]) == 0) {
      linemodrgb = true;
    }
    if (strcmp("--linemod", argv[h]) == 0) {
      linemod = true;
    }
    if (strcmp("--hsv", argv[h]) == 0) {
      hsv = true;
    }
    if (strcmp("--rgb", argv[h]) == 0) {
      rgb = true;
    }
    if (strcmp("--no_border", argv[h]) == 0) {
      only_non_border = true;
    }
    if (strcmp("--check_groups", argv[h]) == 0) {
      check_groups = true;
    }
    if (strcmp("-h", argv[h]) == 0) {
      help = true;
    }

  }

  if (help == true) {
    help_f();
    return 0;
  }

  if (line2d == true && linergb == true) {
    printf(
        "LineRGB and Line2D are exclusive. Please choose only one modality\n");
    printf("use -h option to show help\n");
    return 0;
  } else if (line2d == true) {
    printf("Test will be executed with Line2D\n");
  } else if (linergb == true) {
    printf("Test will be executed with LineRGB\n");
  } else if (linemodrgb == true) {
    printf("Test will be executed with LinemodRGB\n");
  } else if (line2d == false && linergb == false && linemodrgb == false
      && linemod == false) {
    printf(
        "Please specify one modality: \"--linergb\" or \"--line2d\" or \"--linemodrgb\" or \"--linemod\"\n");
    printf("use -h option to show help\n");
    return 0;
  }

  if (rgb == true && hsv == true) {
    printf(
        "\"--rgb\" and \"--hsv\" options are exclusive. Please use only one of them\n");
    return 0;
  }
  if (rgb == false && hsv == false) {
    printf("Please use one between \"--rgb\" and \"--hsv\" option\n");
    return 0;
  }

  if (only_non_border == true)
    printf("Only consider color features not on the object borders\n");

  /////-TRAIN-/////
  Timer total_train_timer;
  total_train_timer.start();

  cv::Ptr<cv::line_rgb::Detector> detector_rgb;
  cv::Ptr<cv::linemod::Detector> detector_line2d;
  cv::Ptr<cv::linemod::Detector> detector_linemod;
  cv::Ptr<cv::line_rgb::Detector> detector_linemodrgb;

  int num_modalities = 0;
  if (linergb == true) {
    detector_rgb = line_rgb::getDefaultLINERGB(hsv);
    num_modalities = (int) detector_rgb->getModalities().size();
  }
  if (line2d == true) {
    detector_line2d = linemod::getDefaultLINE();
    num_modalities = (int) detector_line2d->getModalities().size();
  }
  if (linemod == true) {
    detector_linemod = linemod::getDefaultLINEMOD();
    num_modalities = (int) detector_linemod->getModalities().size();
  }
  if (linemodrgb == true) {
    detector_linemodrgb = line_rgb::getDefaultLINEMODRGB(hsv);
    num_modalities = (int) detector_linemodrgb->getModalities().size();
  }

  vector<TemplatesObject*> all_objects_templates;
  bool with_depth = true;
  if (line2d == true || linergb == true)
    with_depth = false;
  getAllObjectsTemplates(path_templates, all_objects_templates, with_depth);
  int num_objects = all_objects_templates.size();

  // Extract templates

  string filename;
  if (linergb == true)
    filename = "line_rgb_templates.yml";
  if (line2d == true)
    filename = "line_2d_templates.yml";
  if (linemod == true)
    filename = "linemod_templates.yml";
  if (linemodrgb == true)
    filename = "linemod_rgb_templates.yml";

  string pathof_templ_paths = "PATHS_" + filename;

  if (check_groups == true) {
    do_check_groups(linergb, pathof_templ_paths);
    return -1;
  }
  FileStorage templ_paths(pathof_templ_paths, cv::FileStorage::WRITE);

  for (int no = 0; no < num_objects; no++) {

    TemplatesObject* t_object = all_objects_templates[no];
    string class_id = t_object->name;
    int num_images = t_object->images.size();

    cout << "------ Training " << class_id << " with " << num_images
         << "images --------" << endl;

    for (int ni = 0; ni < num_images; ni++) {
      Pose pose = t_object->poses[ni];
      cout << " 		mask: " << t_object->masks[ni] << endl;
      cout << " 		image: " << t_object->images[ni] << endl;
      if (with_depth)
        cout << " 		depth: " << t_object->depths[ni] << endl << endl;
      cout << "     pose: radius: " << pose.radius
          << " - Tx: " << pose.Tx
          << " - Ty: " << pose.Ty
          << " - Tz: " << pose.Tz
          << " - Angle: " << pose.angle
          << endl;
      cv::Mat mask;
      cv::Mat rgb;
      cv::Mat depth;
      mask = cv::imread(t_object->masks[ni], 0);
      rgb = cv::imread(t_object->images[ni], 1);
      if (with_depth) {
        FileStorage fs(t_object->depths[ni], FileStorage::READ);
        fs["depth"] >> depth;
        depth.convertTo(depth, CV_16U);
      }

      cv::Rect bb;
      std::vector<cv::Mat> sourcesTemplate;
      sourcesTemplate.push_back(rgb);
      if (num_modalities == 2)
        sourcesTemplate.push_back(depth);

      Timer single_train_timer;
      single_train_timer.start();
      int template_id;

      if (linergb == true) {
        template_id = detector_rgb->addTemplate(sourcesTemplate,
                                                class_id, mask,
                                                GROUP_SIMILAR_TEMPLATES,
                                                &bb, &pose);
      }
      if (linemodrgb == true) {
        template_id = detector_linemodrgb->addTemplate(sourcesTemplate,
                                                       class_id, mask,
                                                       GROUP_SIMILAR_TEMPLATES,
                                                       &bb, &pose);
      }
      if (line2d == true) {
        template_id = detector_line2d->addTemplate(sourcesTemplate, class_id,
                                                   mask, &bb);
      }
      if (linemod == true) {
        template_id = detector_linemod->addTemplate(sourcesTemplate, class_id,
                                                    mask, &bb);
      }
      ///////////DEBUG/////////////////////////////
      if (DEBUG == true) {

        std::vector<cv::line_rgb::Template> templates;
        int T;
        if (linergb == true) {
          templates = detector_rgb->getTemplates(class_id, template_id);
          T = detector_rgb->getT(0);
        }
        if (linemodrgb == true) {
          templates = detector_linemodrgb->getTemplates(class_id, template_id);
          T = detector_linemodrgb->getT(0);
        }
        drawResponseLineRGB(templates, num_modalities, rgb,
                            cv::Point(bb.x, bb.y), T, -1, class_id,
                            only_non_border);
        /*

         vector <cv::line_rgb::Feature> all_features;
         all_features.insert(all_features.end(), templates[0].features_border.begin(),
         templates[0].features_border.end());
         all_features.insert(all_features.end(), templates[0].features_inside.begin(),
         templates[0].features_inside.end());
         for (int l = 0; l < (int)all_features.size(); ++l)
         {
         cv::Scalar colorT;
         cv::line_rgb::Feature f = all_features[l];
         cv::Point pt(f.x + bb.x, f.y +bb.y);
         switch(f.rgb_label)
         {
         case 0:
         colorT = CV_RGB(255, 0, 0);
         break;
         case 1:
         colorT = CV_RGB(0, 255, 0);
         break;
         case 2:
         colorT = CV_RGB(0, 0, 255);
         break;
         case 3:
         colorT = CV_RGB(255, 255, 0);
         break;
         case 4:
         colorT = CV_RGB(255, 0, 255);
         break;
         case 5:
         colorT = CV_RGB(0, 255, 255);
         break;
         case 6:
         colorT = CV_RGB(255, 255, 255);
         break;
         case 7:
         colorT = CV_RGB(0, 0, 0);
         break;

         }
         //if(f.onBorder == true)
         //rectangle(rgb, bb, CV_RGB(255,255,255), 2);
         cv::circle(rgb, pt, 1, colorT);
         }*/
        imshow("color_rotated_featured", rgb);
        waitKey();
      }
      /////////fine debug//////////////////

      single_train_timer.stop();
      printf("Train single: %.2fs\n", single_train_timer.time());
      printf("\n.....templating...");
      cout << class_id << endl;
      if (template_id != -1) {
        cout << "*** Added template (id " << template_id
             << ") for new object class " << class_id << " - path:"
             << t_object->images[ni] << "***" << endl;
        templ_paths << class_id + "_" + i2s(template_id)
                    << t_object->images[ni];

      }

    }

  }
  //SAVING TEMPLATES YAML
  if (linergb == true)
    writeLineRGB(detector_rgb, filename, hsv);
  if (line2d == true)
    writeLine2D(detector_line2d, filename);
  if (linemod == true)
    writeLine2D(detector_linemod, filename);
  if (linemodrgb == true)
    writeLineRGB(detector_linemodrgb, filename, hsv);
  cout << endl << filename << " saved" << endl;
  templ_paths.release();
  total_train_timer.stop();
  cout << "Whole train time: " << total_train_timer.time() << endl;

  //free memory
  for (int no = 0; no < all_objects_templates.size(); no++)
    delete (all_objects_templates[no]);

  return 0;
}

