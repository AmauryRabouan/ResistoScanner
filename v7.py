#importations
import sys
import cv2 as cv
import numpy as np
import pandas as pd
from os import system
from math import *
from dataclasses import dataclass
import random

#niveau de floutage
MAX_KERNEL_LENGTH = 35

src = None
dst = None
@dataclass
class cou:
    color: str
    n:int
    i:int

#Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('data/colors_mm.csv', names=index, header=None)

def increase_brightness(img, value):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B):
    minimum = 10000
    imin=0
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            imin=i
            cname = csv.loc[i,"color_name"]
    
    ##print(str(csv.loc[imin,"color_name"])+"  "+str(csv.loc[imin,"color"])+"  "+str(R)+" "+str(G)+" "+str(B))
    return cname

#revoie la moyenne de la luminositée de l'image
def moyImgLum(img,height,width):
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    moyenne=[]
    for x in range (height):
        moyenne.append(sum(v[x])/len(v[x]))
    moy=sum(moyenne)/len(moyenne)
    return moy


def SupremeErrorSmasher(couleur,width,taille):
    nbCouleur={}
    nbCouleur["Fond"]=0
    nbCouleur["Black"]=0
    nbCouleur["Red"]=0
    nbCouleur["Green"]=0
    nbCouleur["Brown"]=0
    nbCouleur["Grey"]=0
    nbCouleur["Orange"]=0
    nbCouleur["Yellow"]=0
    nbCouleur["White"]=0
    nbCouleur["Purple"]=0
    nbCouleur["Blue"]=0
    for i in range(len(couleur)):
        nbCouleur[couleur[i].color]=nbCouleur[couleur[i].color]+couleur[i].n

    maxi=0
    coul="Fond"
    for c,n in nbCouleur.items():
        if(maxi<n):
            maxi=n
            coul=c
    ##print(str(coul)+"  "+str(maxi))
    for j in range(len(couleur)):
        if(couleur[j].color==coul):
            couleur[j].color="Fond"
    ##print(nbCouleur)
    var=len(couleur)-1
    for i in range(var):
        if(i<var):
            if(couleur[i].color==couleur[i+1].color):
                couleur[i].n=couleur[i].n+couleur[i+1].n
                del(couleur[i+1])
                var=var-1

    
    

    miniw=width*0.01
    if(miniw<2):
        miniw=2
    ##print(miniw)
   
    if(couleur[0].n<miniw):
        couleur[1].n=couleur[1].n+couleur[0].n
        del(couleur[0])
    if(couleur[len(couleur)-1].n<miniw):
        couleur[len(couleur)-2].n=couleur[len(couleur)-2].n+couleur[len(couleur)-1].n
        del(couleur[len(couleur)-1])
        
    for j in range(len(couleur)):
        for i in range(len(couleur)):
            ##print(couleur)
            if(i>0 and i<len(couleur)-1):
                if(couleur[i].color=="Fond" and couleur[i].n<miniw*2):
                    if(couleur[i+1].n>=couleur[i-1].n):
                        couleur[i+1].n=couleur[i+1].n+couleur[i].n
                        del(couleur[i])
                        ##print(6)
                        continue
                    else:
                        couleur[i-1].n=couleur[i-1].n+couleur[i].n
                        del(couleur[i])
                        ##print(7)
                        continue
                if(couleur[i].n<miniw):
                    ##print(couleur[i])
                    if(couleur[i-1].color=="Fond" and couleur[i+1].color=="Fond"):
                        couleur[i].n=couleur[i].n+couleur[i+1].n+couleur[i-1].n
                        couleur[i].color="Fond"
                        del(couleur[i-1])
                        del(couleur[i])
                        ##print(1)
                        continue
                    if(couleur[i-1].color=="Fond" and couleur[i+1].color!="Fond"):
                        couleur[i+1].n=couleur[i+1].n+couleur[i].n
                        del(couleur[i])
                        ##print(2)
                        continue
                    if(couleur[i+1].color=="Fond" and couleur[i-1].color!="Fond"):
                        couleur[i-1].n=couleur[i-1].n+couleur[i].n
                        ##print(3)
                        del(couleur[i])
                        continue
                    if(couleur[i+1].color!="Fond" and couleur[i-1].color!="Fond"):
                        if(couleur[i+1].n>=couleur[i-1].n):
                            couleur[i+1].n=couleur[i+1].n+couleur[i].n
                            del(couleur[i])
                            ##print(4)
                        else:
                            couleur[i-1].n=couleur[i-1].n+couleur[i].n
                            del(couleur[i])
                            ##print(5)
                        
    ##print(couleur)
    var=len(couleur)-1
    for i in range(var):
        if(i<var):
            if(couleur[i].color==couleur[i+1].color):
                couleur[i].n=couleur[i].n+couleur[i+1].n
                del(couleur[i+1])
                var=var-1

    if(couleur[0].color=="Grey" or couleur[0].color=="Black"):
        couleur[0].color="Fond"

    if(couleur[len(couleur)-1].color=="Grey" or couleur[len(couleur)-1].color=="Black"):
        couleur[len(couleur)-1].color="Fond"
    
    for i in range(len(couleur)):
        if(i>1 and i<len(couleur)):
            if(couleur[i].color=="Grey"):
                if(couleur[i-1].color=="Black"):
                    couleur[i-i].n=couleur[i-i].n+couleur[i].n
                    del(couleur[i])
        if(i<len(couleur)-1):
            if(couleur[i].color=="Grey"):
                if(couleur[i+1].color=="Black"):
                    couleur[i+1].n=couleur[i+1].n+couleur[i].n
                    del(couleur[i])
        if(i>1 and i<len(couleur)):
            if(couleur[i].color=="Red"):
                if(couleur[i-1].color=="Orange"):
                    if(couleur[i].n>=couleur[i-1].n):
                        couleur[i].n=couleur[i-1].n+couleur[i].n
                        del(couleur[i-1])
                    else:
                        couleur[i].n=couleur[i-1].n+couleur[i].n
                        couleur[i].color=couleur[i-1].color
                        del(couleur[i-1])
        if(i<len(couleur)-1):
            if(couleur[i].color=="Red"):
                if(couleur[i+1].color=="Orange"):
                    if(couleur[i].n>=couleur[i+1].n):
                        couleur[i].n=couleur[i+1].n+couleur[i].n
                        del(couleur[i+1])
                    else:
                        couleur[i].n=couleur[i+1].n+couleur[i].n
                        couleur[i].color=couleur[i+1].color
                        del(couleur[i+1])

    var=len(couleur)-1
    for i in range(var):
        if(i<var):
            if(couleur[i].color==couleur[i+1].color):
                couleur[i].n=couleur[i].n+couleur[i+1].n
                del(couleur[i+1])
                var=var-1
        
    if(couleur[0].color !="Fond" and couleur[len(couleur)-1].color !="Fond"):            
        mini=0
        avant=1000
        apres=1000
        for i in range(len(couleur)):
            if(couleur[i].color!="Fond"):
                if(i>1):
                    #print("CompteurAv:")
                    #print(couleur[i-1].n)
                    avant=couleur[i-1].n
                if(i<len(couleur)-2):
                    #print("CompteurAp:")
                    #print(couleur[i+1].n)
                    apres=couleur[i+1].n
                if(avant<apres):
                    mini=avant
                else:
                    mini=apres
                if(mini>width*0.2):
                    couleur[i].color="Fond"

    var=len(couleur)-1
    for i in range(var):
        if(i<var):
            if(couleur[i].color==couleur[i+1].color):
                couleur[i].n=couleur[i].n+couleur[i+1].n
                del(couleur[i+1])
                var=var-1        

    for i in range(len(couleur)): 
        if(couleur[i].color=="Green" or couleur[i].color=="Yellow" or couleur[i].color=="Purple" or couleur[i].color=="Orange"or couleur[i].color=="Blue"or couleur[i].color=="Red" and couleur[i].n > miniw*2):
            couleur[i].n=couleur[i].n+6
    for i in range(len(couleur)):   
        if(i>0 and i<len(couleur)-1):
            if(couleur[i].color=="Grey" or couleur[i].color=="Orange"):
                if(couleur[i-1].color!="Fond" and couleur[i+1].color!="Fond"):
                    couleur[i].color="Fond"

    for i in range(len(couleur)-1):
        if(i<len(couleur)-2):
            if(couleur[i].color!="Fond" and couleur[i+1].color!="Fond"):
                if(couleur[i].n>couleur[i+1].n):
                    couleur[i].n=couleur[i].n+couleur[i+1].n
                    del(couleur[i+1])
                else:
                    couleur[i].n=couleur[i].n+couleur[i+1].n
                    couleur[i].color=couleur[i+1].color
                    del(couleur[i+1])
    if(couleur[0].color=="Fond" and couleur[0].n<4):
        if(couleur[1].color =="Black"or couleur[1].color=="Brown" or couleur[1].color== "Grey"):
            couleur[1].color="Fond"
            couleur[1].n=couleur[1].n+couleur[0].n
            del(couleur[0])
    var=len(couleur)-1
    for i in range(var):
        if(i<var):
            if(couleur[i].color==couleur[i+1].color):
                couleur[i].n=couleur[i].n+couleur[i+1].n
                del(couleur[i+1])
                var=var-1       
    #print(couleur)

    return couleur

#function to calculate minimum distance from all colors and get the most matching color
def rainbowCompactor(couleur,taille):
    fond=0
    fonde=0
    poscouleur=[]
    somme=0
    for i in range(len(couleur)):
        if(couleur[i].color=='Fond'):
            if(fond==0):
                fond=1
                fonds=i
            if(fond==1):
                fonde=i
        poscouleur.append(somme)
        somme=somme+couleur[i].n
    fond=2
    ##print(fonde)
    maxc=[]
    for u in range(taille):
        maxc.append(cou("",0,u))
        maxn=0
        maxi=0
        iii=0
        ##print(i)
        for i in range(len(couleur)):
            if(maxn<couleur[i].n and couleur[i].color!='Fond'):
                maxn=couleur[i].n
                maxi=couleur[i].i
                iii=i
        try:
            maxc[u].color=couleur[iii].color
        except:
            res=[1]
            return res
        maxc[u].n=couleur[iii].n
        maxc[u].i=maxi
        del(couleur[iii])
    
    for i in range(len(maxc)-1):
        for j in range(len(maxc)-1):
            if(maxc[j].i>maxc[j+1].i):
                tmpc=maxc[j].color
                tmpn=maxc[j].n
                tmpi=maxc[j].i
                maxc[j].color=maxc[j+1].color
                maxc[j].n=maxc[j+1].n
                maxc[j].i=maxc[j+1].i
                maxc[j+1].color=tmpc
                maxc[j+1].n=tmpn
                maxc[j+1].i=tmpi
    moy=0
    somme2=0
    somme3=0
    for i in range(len(couleur)):
        if(couleur[i].i<maxc[0].i):
            somme2=somme2+couleur[i].n
        if(couleur[i].i>maxc[taille-1].i):
            somme3=somme3+couleur[i].n
    #print("#####"+str(somme2)+"#####"+str(somme3)+"#####")
    if(somme2>somme3):
        maxc.reverse()
    return maxc

#fonction de calcul d'un angle avec 3 vecteurs
def calcul_ange(x1,y1,x2,y2,x3,y3):
    numerateur=y2*(x1-x3)+y1*(x3-x2)+y3*(x2-x1)
    denominateur=(x2-x1)*(x1-x3)+(y2-y1)*(y1-y3)
    if(denominateur==0):
        return 0
    ratio = numerateur/denominateur
    angleRad = atan(ratio)
    angleDeg = (angleRad*180)/pi
    if(angleDeg<0):
        angleDeg=180+angleDeg
    return angleDeg

#fonction de rotation d'image
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


def main(argv):
    system('cls')
    print (cv.__version__)
    print("|----------|")
    # Load the source image
    imageName = argv[0]

    global src
    src = cv.imread(imageName)
    src = cv.detailEnhance(src, 10, 0.15)

    height, width = src.shape[:2]
   
    #changement de resolution
    if(height>width) :
        var = 1200/height*100
    if(height<=width) :
        var = 1200/width*100
    scale_percent = var # percent of original size
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)
    system('cls')
    print("|█---------|")
    # resize image
    src = cv.resize(src, dim, interpolation = cv.INTER_AREA)
    if src is None:
        #print ('Error opening image')
        return -1
    
    #Affichage image 
    #cv.imshow('poischiche',src) 
    #cv.waitKey()
    #cv.destroyAllWindows()
   
    moy=moyImgLum(src,height,width)
    #print(moy)
    rectif=236-moy
    src=increase_brightness(src,(int)(rectif))
    system('cls')
    print("|██--------|")
    #moy=moyImgLum(src,height,width)
    #print(moy)
    #Affichage image 
    #cv.imshow('poischiche',src) 
    #cv.waitKey()
    #cv.destroyAllWindows()
    global dst 
    dst = np.copy(src)
    dst2 = np.copy(src)

    

    # Applying Median blur entre 5 et 8
    for i in range(1, (int)(MAX_KERNEL_LENGTH), 2):
        dst = cv.medianBlur(src, i)

    #Affichage image 
    #cv.imshow('poischiche',dst) 
    #cv.waitKey()
    #cv.destroyAllWindows()

    
    #HSV
    hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV) #image en nuances de gris
    h, s, v = cv.split(hsv)
    #Affichage image 
    #cv.imshow('poischiche',s) 
    #cv.waitKey()
    #cv.destroyAllWindows()

    edges = cv.Canny(s,20,130,apertureSize = 3) #image noir et contour blanc

    #Affichage image 
    #cv.imshow('poischiche',edges) 
    #cv.waitKey()
    #cv.destroyAllWindows()
    system('cls')
    print("|███-------|")
    #classe pour découper les resistances
    @dataclass
    class boite:
        x: int
        y: int
        w: int
        h: int
        verif: int
    
    #recherche de contours
    contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)# La fonction findContours nous retourne les contours des objets présents dans une image binaire, elle prend en paramètre une image binaire, le mode, qui indique à la fonction les contours dont nous souhaitons obtenir cela peut être les contours extérieurs, intérieurs ou les deux. Enfin, l’argument method indique comment nous souhaitons que les contours soient représentés dans notre cas, ils seront représentés par une suite de points connectés.
    bo=[0]
    bo[0]=0
    for i in range (0, len(contours)) :
        mask_BB_i = np.zeros((len(edges),len(edges[0])), np.uint8)
        x,y,w,h = cv.boundingRect(contours[i])#Pour finir boundingrect est une fonction qui retourne les coordonnées de la boundingbox d’un contour
        cv.drawContours(mask_BB_i, contours, i, (255,255,255), -1)#DrawContours va permettre de dessiner un à un sur des images vierges  chacun des contours
        BB_i=cv.bitwise_and(dst,dst,mask=mask_BB_i) # opperateur and sur deux images en fonction des blancs
        if h >15 and w>15 :
                bo.append(boite(x,y,w,h,1))
    system('cls')
    print("|████------|")
    #verification si les resistances se superposent
    for i in range(1,len(bo)):
        for j in range(1,len(bo)):
            if(i!=j and bo[j].x>=bo[i].x and (bo[j].x+bo[j].w)<=(bo[i].x+bo[i].w) and bo[j].y>=bo[i].y and (bo[j].y+bo[j].h)<=(bo[i].y+bo[i].h) and bo[i].verif==1):
                bo[j].verif=0
    ok=[0]
    okpos=[0]
    ok[0]=0
    compteur=0
    rec=dst2
    #enregistrement des petites images
    for i in range (1, len(bo)) :
        if(bo[i].verif==1):
            compteur= compteur+1
            y=bo[i].y
            x=bo[i].x
            h=bo[i].h
            w=bo[i].w
            partie=src[y:y+h,x:x+w]

            #eviter l'erreur quand on prend une partie sur le bord de l'image
            
         
            if(i==1):
                ok[0]=partie
                okpos[0]=x
                okpos.append(y)
            else:
                ok.append(partie)
                okpos.append(x)
                okpos.append(y)
            
            start_point =(x,y) #coordonnées pour le rectangle
            end_point = (x+w,y+h)
            color = (255, 0, 0)
            thickness = 2
            cv.rectangle(rec, start_point, end_point, color, thickness)#affichage rectangle 

    #Affichage image 
    #cv.imshow('poischiche',rec) 
    #cv.waitKey()
    #cv.destroyAllWindows()
    system('cls')
    print("|█████-----|")
    count = 0
    count2 = 0
    i=0
    for img in ok :
        lines=0
        #Affichage image 
        #cv.imshow('poischiche',img) 
        #cv.waitKey()
        #cv.destroyAllWindows()
        img2=np.copy(img)
        for i in range(1, 1, 2):
            img2 = cv.medianBlur(img2, i)
         #eviter erreur
        try:
            grayImage = cv.cvtColor(img2, cv.COLOR_BGR2HSV) #image en Hsv
             
        except :
            del ok[i]
            count2 = count2 +1
            print("ayaya")
            continue
        i=i+1
        edges = cv.Canny(grayImage,50,200,apertureSize = 3) #image noir et contour blanc
        #Affichage image 
        #cv.imshow('poischiche',edges) 
        #cv.waitKey()
        #cv.destroyAllWindows()
        lines = cv.HoughLines(edges,1,np.pi/180,30)#trouver des lignes sur l'image
        moyx1=0
        moyy1=0
        moyx2=0
        moyy2=0
        moya=0
        if lines is None :
           continue
        for i in range(len(lines)):
            for rho,theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                #calcul angles aux croisements des lignes
                f=x2-x1
                g=y2-y1
                u = sqrt(f*f+g*g)
                v = 1
                uv = y1
                cos = uv/u*v

                #cv.line(img,(x1,y1),(x2,y2),(255, 0, 0),2)
                 
                moya=cos+moya
                moyx1=x1+moyx1
                moyx2=x2+moyx2
                moyy1=y1+moyy1
                moyy2=y2+moyy2


        moya=moya/len(lines)
        moyx1=int(moyx1/len(lines))
        moyx2=int(moyx2/len(lines))
        moyy1=int(moyy1/len(lines))
        moyy2=int(moyy2/len(lines))
        #cv.line(img,(moyx1,moyy1),(moyx2,moyy2),(1, 1, 254),2)
        angle=calcul_ange(moyx2,moyy2,moyx1,moyy1,moyx1,moyy2)
        img=rotate_bound(img,-angle)
        (h, w) = img.shape[:2]
        for x in range(w) :
            for y in range(h):
                if(img[y][x][0]==0):
                    img[y][x]=[255,255,255]
        ok[count]=img
        count=count+1
        system('cls')
        print("|█████-----|")
        print("|",end="")
        for i in range(count):
            print("•",end="")
        for i in range(len(ok)-count):
            print(" ",end="")
        print("|")
        #affichage image
        #cv.imshow('poischiche',img) 
        #cv.waitKey()
        #cv.destroyAllWindows()
    


    system('cls')
    print("|████████--|")
    comptimg=0
    resistor=1
    for img in ok :
    
    #detection des couleurs de la resistances 
        
        img2 = np.copy(img)
        height, width = img.shape[:2]
        xc=0
        nbcolor=-1
        #cv.imshow('recadrage',img2) 
        #cv.waitKey()
        #cv.destroyAllWindows()

        lastcolor='None'
        couleur=[]
        for x in img[(int)(height/2)] :
            ##print(x)
            taille =10
            #visuel en haut avec la moyenne du milieu de la résistance 
            for y in range ((int)(height-(height*70/100))) :
                r=0
                g=0
                b=0
                for yc in range (taille): 
                    r=r+img[(int)(height/2-(taille/2)+yc)][xc][2]
                    g=g+img[(int)(height/2-(taille/2)+yc)][xc][1]
                    b=b+img[(int)(height/2-(taille/2)+yc)][xc][0]
                r=r/taille
                ##print(r)
                g=g/taille
                ##print(g)
                b=b/taille
                ##print(b)
                img2[y][xc]=[(int)(b),(int)(g),(int)(r)]  
            color=getColorName((int)(r),(int)(g),(int)(b))
            rgbp = np.uint8([[[img2[5][xc][0],img2[5][xc][1],img2[5][xc][2] ]]])
            r=255
            g=255
            b=255
            ##print(color)
            if(color=='Red'):
                ##print(color)
                r=255
                g=0
                b=0
                if(lastcolor=='Red'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Red'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Red",1,xc))
                lastcolor='Red'
            if(color=='Blue'):
                ##print(color)
                r=0
                g=0
                b=255
                if(lastcolor=='Blue'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Blue'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Blue",1,xc))
                lastcolor='Blue'
            if(color=='Green'):
                ##print(color)
                r=0
                g=255
                b=0
                if(lastcolor=='Green'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Green'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Green",1,xc))
                lastcolor='Green'
            if(color=='Black'):
                ##print(color)
                r=0
                g=0
                b=0
                if(lastcolor=='Black'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Black'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Black",1,xc))
                lastcolor='Black'
            if(color=='Brown'):
                ##print(color)
                r=77
                g=63
                b=48
                if(lastcolor=='Brown'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Brown'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Brown",1,xc))
                lastcolor='Brown'
            if(color=='Yellow'):
                ##print(color)
                r=240
                g=242
                b=80
                if(lastcolor=='Yellow'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Yellow'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Yellow",1,xc))
                lastcolor='Yellow'
            if(color=='Orange'):
                ##print(color)
                r=214
                g=130
                b=41
                if(lastcolor=='Orange'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Orange'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Orange",1,xc))
                lastcolor='Orange'
            if(color=='Grey'):
                ##print(color)
                r=143
                g=141
                b=137
                if(lastcolor=='Grey'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Grey'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Grey",1,xc))
                lastcolor='Grey'
            if(color=='Purple'):
                ##print(color)
                r=143
                g=12
                b=237
                if(lastcolor=='Purple'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Purple'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Purple",1,xc))
                lastcolor='Purple'
            if(color=='Fond'):
                r=163
                g=142
                b=108
                if(lastcolor=='Fond'):
                    couleur[nbcolor].n=couleur[nbcolor].n+1
                if(lastcolor!='Fond'):
                    nbcolor=nbcolor+1
                    couleur.append(cou("Fond",1,xc))
                lastcolor='Fond'

            #visuel des couleur trouvées
            for y in range ((int)(height-(height*70/100))) :
               img2[y+(int)(height*70/100)][xc]=[b,g,r]
            xc=xc+1
            #affichage image   
            #cv.imshow('poischiche',img2)  
            #cv.waitKey()
            #cv.destroyAllWindows()
        
        #print("####################################################") 
        ##print(couleur)
        SupremeErrorSmasher(couleur,width,3)
        #print(couleur)
        res=rainbowCompactor(couleur, 3)
        if(len(res)!=3):
            continue
        #for i in range(len(res)):
            #print(res[i].color)
        #print("####################################################") 
        ohm = 0

        bande = {
            'Black': 0, 
            'Brown': 1,
            'Red': 2,
            'Orange': 3,
            'Yellow': 4,
            'Green': 5,
            'Blue': 6,
            'Purple': 7,
            'Grey': 8,
            'White': 9,
            'Gold': 0.1,
            'Sliver': 0.01,
        }

        ohm = bande.get(res[0].color, 0)
        ohm = ohm*10
        ohm = ohm+bande.get(res[1].color, 0)
        ohm = ohm*pow(10, bande.get(res[2].color, 0))

        #print(ohm)
        font = cv.FONT_HERSHEY_SIMPLEX
        if(ohm>0 and ohm <1000):
            cv.putText(src,str(ohm)+" Ohm",(okpos[comptimg-count2*2],okpos[comptimg+1-count2*2]),font,1,(0,0,0),2)
        if(ohm>=1000 and ohm <1000000):
            ohm=ohm/1000
            cv.putText(src,str(ohm)+" KOhm",(okpos[comptimg-count2*2],okpos[comptimg+1-count2*2]),font,1,(0,0,0),2)
        if(ohm>=1000000 and ohm <1000000000):
            ohm=ohm/1000000
            cv.putText(src,str(ohm)+" MOhm",(okpos[comptimg-count2*2],okpos[comptimg+1-count2*2]),font,1,(0,0,0),2)
        if(ohm>=1000000000 and ohm <1000000000000):
            ohm=ohm/1000000000
            cv.putText(src,str(ohm)+" GOhm",(okpos[comptimg-count2*2],okpos[comptimg+1-count2*2]),font,1,(0,0,0),2)
        comptimg=comptimg+2
        #affichage image   
        #cv.imshow('poischiche',img2)  
        #cv.waitKey()
        #cv.destroyAllWindows()
        system('cls')
        print("|████████--|")
        print("|",end="")
        for i in range(resistor):
            print("•",end="")
        for i in range(len(ok)-resistor):
            print(" ",end="")
        print("|")
        resistor=resistor+1
    system('cls')
    print("|██████████|")
    print("finished")
    cv.imwrite("resultat.png", src)
    #affichage image   
    cv.imshow('final',src)  
    cv.waitKey()
    cv.destroyAllWindows()
       
            

if __name__ == "__main__": 
    main(sys.argv[1:])
