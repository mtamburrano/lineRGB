//      generate_similarity_HSV.cpp
//
//      Copyright 2012 Pompolus <pompolus@pompolus-laptop>
//
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.


#include <iterator>
#include <set>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;


static unsigned char tableHSV[8][8];

void import_csv()
{

    ifstream in("tabella_senza_margini_main.csv");
    //ifstream in("tabella_senza_margini_compatibile.csv");

    string line, field;

    int r = 0, c = 0;
    while ( getline(in,line) )    // get next line in file
    {
        c = 0;
        stringstream ss(line);

        while (getline(ss,field,','))  // break line into comma delimitted fields
        {
            int number;
            ss << field;
            ss >> number;
            tableHSV[r][c] = number;  // add each field to the 1D array
            c++;
        }

        r++;
    }

    for (size_t i=0; i<8; ++i)
        {
            cout<<i<<": |";
            for (size_t j=0; j<8; ++j)
            {
                cout << (int)tableHSV[i][j] << "|"; // (separate fields by |)
            }
            cout << "\n";
        }

    //std::cout<<"tableHSV[128][128]: "<<tableHSV[0][128]<<std::endl;
}




static inline int getLabel(int quantized) {
    switch (quantized) {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    case 256:
        return 8;
    case 512:
        return 9;
    case 1024:
        return 10;
    case 2048:
        return 11;
    case 4096:
        return 12;
    case 8192:
        return 13;
    case 16384:
        return 14;
    case 32768:
        return 15;
    default:
        cout<<"errore"<<endl;
        return -1; //avoid warning
    }
}

unsigned char calc1istance (int number, int ori)
{

    int b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16;

    b1 = number & 1;
    b2 = number & 2;
    b3 = number & 4;
    b4 = number & 8;
    b5 = number & 16;
    b6 = number & 32;
    b7 = number & 64;
    b8 = number & 128;
    b9 = number & 256;
    b10 = number & 512;
    b11 = number & 1024;
    b12 = number & 2048;
    b13 = number & 4096;
    b14 = number & 8192;
    b15 = number & 16384;
    b16 = number & 32768;



    unsigned char best_ori_value = 0;

     if((int)b1 != 0)
     {
        int tmpValue = getLabel((int)b1);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b2 != 0)
     {
        int tmpValue = getLabel((int)b2);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b3 != 0)
     {
        int tmpValue = getLabel((int)b3);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b4 != 0)
     {
        int tmpValue = getLabel((int)b4);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b5 != 0)
     {
        int tmpValue = getLabel((int)b5);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b6 != 0)
     {
        int tmpValue = getLabel((int)b6);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b7!= 0)
     {
        int tmpValue = getLabel((int)b7);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b8 != 0)
     {
        int tmpValue = getLabel((int)b8);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b9 != 0)
     {
        int tmpValue = getLabel((int)b9);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b10 != 0)
     {
        int tmpValue = getLabel((int)b10);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b11 != 0)
     {
        int tmpValue = getLabel((int)b11);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b12 != 0)
     {
        int tmpValue = getLabel((int)b12);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b13 != 0)
     {
        int tmpValue = getLabel((int)b13);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b14 != 0)
     {
        int tmpValue = getLabel((int)b14);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b15!= 0)
     {
        int tmpValue = getLabel((int)b15);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }
     if((int)b16 != 0)
     {
        int tmpValue = getLabel((int)b16);
        if(tableHSV[tmpValue][ori] > best_ori_value)
            best_ori_value = tableHSV[tmpValue][ori];
     }

     return best_ori_value;
}


int main(int argc, char** argv)
{
    import_csv();

    ofstream myfile;
    myfile.open ("similarity_HSV.csv");
    //myfile.open ("similarity_HSV_compatibile.csv");

    int similarity_HSV[8][256];

    for(int number = 0; number <256; number++)
    {
        for(int ori = 0; ori < 8; ori++)
        {
            similarity_HSV[ori][number] = calc1istance(number,ori);
        }
    }

    for(int ori = 0; ori<8; ori++)
    {
        myfile << ",";
        for(int number= 0; number<256; number++)
        {
            myfile << similarity_HSV[ori][number];
            if(number != 255)
                myfile << ", ";
        }
        myfile << endl;
    }

    myfile.close();

//  cout<<"SIMILARITY_HSV_LUT[16][32768] = {";
//
//      for(int ori = 0; ori<16; ori++)
//      {
//          cout<<"{";
//          for(int number= 0; number<32768; number++)
//          {
//              cout<<similarity_HSV[ori][number];
//              if(number != 32767)
//                  cout<<", ";
//          }
//          cout<<"}";
//          if(ori != 15)
//              cout<<", ";
//      }
//
//      cout<<"};";



    return 0;
}
