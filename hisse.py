
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# yfinance ile veriler internetten çekilir.
!pip uninstall -y yfinance
pip install yfinance
import yfinance as yf

#Kullanıcıdan Hisse adı ve bakılması istenen başlangıç ve bitiş tarihi aralığı
tic = input("Öğrenmek istediğiniz hisse adı: ")
start = input ("Başlangıç tarihi girin (yyyy-mm-dd): ")
end = input ("Bitiş tarihi girin (yyyy-mm-dd): ")
data = yf.download(tic , start=start, end=end, progress=False)

#Açılış, Kapanış, en yüksek, en düşük bilgiler gösterilir. Kapanış fiyatları yazdırıyoruz.
datacl=data.filter(['Close'])

#Kapanış fiyatı grafiği
datacl ['Close'].plot(figsize = (25,12))
plt.title ('Fiyatlar')
plt.plot(data['Close'])
plt.show()

#Kullanıcıya 30 günlük veri tahmini yapacağımızı söyledik. Close fiyatlarını tahmin yapacağımız güne kaydırdık. 
day = 30
datacl ['Tahmin'] = datacl[['Close']].shift (-day)

#Tahmin sutununun 1.satırı yazılır
x = np.array (datacl.drop (['Tahmin'] , 1 ))[:-day]
print (x)

y = np.array (datacl.drop (['Tahmin'] , 1 ))[:-day]
print (y)

#Toplam 36 tane verimiz var. 10 verimizin test olmasını istedik.  
x_train , x_test, y_train , y_test = train_test_split(x,y,test_size = 10/36 )

tree = DecisionTreeRegressor ().fit (x_train , y_train)
lr = LinearRegression().fit(x_train , y_train)

#Tahminimizdeki verileri array olarak yazdırıyoruz.
x_day = datacl.drop (['Tahmin'],1)[-day : ]
x_day = x_day.tail(day)
x_day = np.array(x_day)

tree = tree.predict (x_day)
print (tree )

##predict() her test örneği için bir tahmin gerçekleştirir
lr = lr.predict(x_day)
print (lr)

# Model oluşur
pre = tree
valid = datacl[x.shape[0]:]
valid['Pre'] = pre
plt.figure(figsize=(25,12))
plt.title('Model')
plt.xlabel ('Günler')
plt.ylabel(str(tic)+' Fiyat')
plt.plot(datacl['Close'])
plt.plot(valid[['Close' , 'Pre']])
plt.legend (['Orijinal', 'Değer' , 'Pre'])
plt.show()

# Model oluşur
pre = lr
valid = datacl[x.shape[0]:]
valid['Pre'] = pre
plt.figure(figsize=(25,12))
plt.title('Model')
plt.xlabel ('Günler')
plt.ylabel( str(tic)+' Fiyat' )
plt.plot(datacl['Close'])
plt.plot(valid[['Close' , 'Pre']])
plt.legend (['Orijinal', 'Değer' , 'Pre'])
plt.show()


