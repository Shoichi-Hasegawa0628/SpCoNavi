#!/usr/bin/env python
#-*- coding:utf-8 -*-

import urllib
import urllib2
import sys
import numpy as np
import os
import cv2
import socket
import httplib
def download():
    os.mkdir(sys.argv[2])
    f=open(sys.argv[1])
    lines = f.readlines()
    print lines[0]
    f.close()
    i=0
    for l in lines:
        url = l
        print url
        title = sys.argv[2]+"/"+sys.argv[2]+repr(i)+".jpg"
        try:
            resp = urllib2.urlopen(url)
            urllib.urlretrieve(url,"{0}".format(title))
            i +=1
            im=cv2.imread(title)
            if im[0,0][0]==238&im[0,0][1]==238&im[0,0][2]==238:
                os.remove(title)
                i -=1


            if i==240:
                sys.exit()
        except urllib2.HTTPError:
            pass
        except ValueError: 
            pass
        except urllib2.URLError: 
            pass
        except socket.error:
            pass
        except TypeError:
            i-=1
            pass
        except httplib.BadStatusLine:
            pass
        except NameError:
            pass
if __name__ == "__main__":

    download()