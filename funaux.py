# version 0.1.3 16/6/2023
print("importando modulos de ploteo, audio e interaccion")
   
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import scipy.fftpack as fft
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

print("definiendo funciones auxiliares")

def normalizar(x):
  '''
    reescala los valores de entrada de manera que los valores
    de salida esten entre 0 y 1
  '''
  xn=x-min(x)
  xn=xn/max(xn)
  return xn


def normalizarb(x):
  '''
     reescala los valores de entrada de manera que los
     valores de salida esten entre -1 y 1
  '''
  xn=2*normalizar(x)-1
  return xn


def graft(x,fs,xlim):
  '''
     grafica en el dominio del tiempo
     primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo
     tercer argumento, intervalo de tiempo a graficar [tmin, tmax]
  '''
  tmax = len(x)/fs
  t = np.arange(0, tmax, step=1. / fs)
  plt.plot(t, x)
  if len(xlim) >= 2:
      plt.xlim(xlim) 
  plt.xlabel("Tiempo (s)")
  plt.ylabel("Amplitud")
  return

def graff1l(x,fs,xlim):
  '''
     grafica en el dominio de la frecuencia, espectro de un lado
       primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo
     tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
     entrega componentes en frecuencia (nros. complejos)
  '''
  N = len(x)
  X=np.fft.fft(x)/N
  freqs = np.arange(0, fs / 2, step=fs / N)
  plt.plot(freqs[:(N // 2)], 2*np.abs(X[:(N // 2)]))
  plt.xlabel("Frequencia (Hz)")
  plt.ylabel("Amplitud")
  if len(xlim) >= 2:
    plt.xlim(xlim) 
  return X

def graff1lf(x,fs,xlim):
  '''
     grafica en el dominio de la frecuencia, espectro de fases de un lado
     primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo
     tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
     entrega componentes en frecuencia (nros. complejos)
  '''
  N = len(x)
  X=np.fft.fft(x)/N
  freqs = np.arange(0, fs / 2, step=fs / N)
  plt.plot(freqs[:(N // 2)], np.angle(X[:(N // 2)],deg=True),'b*')
  plt.xlabel("Frequencia (Hz)")
  plt.ylabel("Amplitud")
  plt.ylim([-180,180]) 
  if len(xlim) >= 2:
      plt.xlim(xlim) 
  return X

def graff2lf(x,fs,xlim):
  '''
     grafica en el dominio de la frecuencia, espectro de fases de un lado
     primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo
     tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
     entrega componentes en frecuencia (nros. complejos)
  '''
  N = len(x)
  X=np.fft.fft(x)/N
  # Plot the positive frequencies.
  freqsp = np.arange(0, fs / 2, step=fs / N)
  plt.plot(freqsp, np.angle(X[:(N // 2)],deg=True),'b*')
  # Plot the negative frequencies.
  freqsn = np.arange(-fs / 2, 0, step=fs / N)
  plt.plot(freqsn, np.angle(X[(N // 2):],deg=True),'b*')
  plt.xlabel("Frequencia (Hz)")
  plt.ylabel("Amplitud")
  plt.ylim([-180,180]) 
  if len(xlim) >= 2:
      plt.xlim(xlim) 
  return X

def graff2l(x,fs,xlim):
  '''
     grafica en el dominio de la frecuencia, espectro de dos lados
     primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo
     tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
     entrega componentes en frecuencia (nros. complejos)
  '''
  N = len(x)
  X = np.fft.fft(x)/N
  # Plot the positive frequencies.
  freqsp = np.arange(0, fs / 2, step=fs / N)
  plt.plot(freqsp, np.abs(X[:(N // 2)]))
  # Plot the negative frequencies.
  freqsn = np.arange(-fs / 2, 0, step=fs / N)
  plt.plot(freqsn, np.abs(X[(N // 2):]))
  # Now we can label the x-axis.
  plt.xlabel("Frequencia (Hz)")
  plt.ylabel("Amplitud")
  if len(xlim) >= 2:
    plt.xlim(xlim) 
  return X

def graff1lp(x,fs,xlim):
  '''
     grafica en el dominio de la frecuencia, espectro de un lado de potencias
     primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo
     tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
     entrega componentes en frecuencia (nros. complejos)
  '''
  N = len(x)
  X=np.fft.fft(x)/N
  X_p = np.abs(X) ** 2
  freqs = np.arange(0, fs / 2, step=fs / N)
  plt.plot(freqs[:(N // 2)], (X_p[:(N // 2)]))
  plt.xlim(xlim)
  plt.xlabel("Frequencia (Hz)")
  plt.ylabel("Potencia")
  if len(xlim) >= 2:
    plt.xlim(xlim) 
  return X

def graff2lp(x,fs,xlim):
  '''
     grafica en el dominio de la frecuencia, espectro de dos lados de potencias
     primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo}
     tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
  '''
  N = len(x)
  X = np.fft.fft(x)/N
  X_p = np.abs(X)**2
  # Plot the positive frequencies.
  freqsp = np.arange(0, fs / 2, step=fs / N)
  plt.plot(freqsp, X_p[:(N // 2)])
  # Plot the negative frequencies.
  freqsn = np.arange(-fs / 2, 0, step=fs / N)
  plt.plot(freqsn, X_p[(N // 2):])
  #  etiquetado de ejes
  plt.xlabel("Frequencia (Hz)")
  plt.ylabel("Amplitud ")
  if len(xlim) >= 2:
    plt.xlim(xlim) 
  return X

def espectro(x,fs,xlim):
  '''
     grafica en el dominio de la frecuencia, espectro de dos lados de potencias en escala logaritmica
     primer argumento, arreglo de valores de la funcion en el tiempo
     segundo argumento, frecuencia de muestreo
     tercer argumento, limites de frecuencias a graficar [finf, fsup]
     entrega componentes en frecuencia (nros. complejos)
  '''
  N = len(x)
  X = np.fft.fft(x)/N
  X_pl = np.log(np.abs(X)**2)
  # plotear frecuencias positivas
  freqsp = np.arange(0, fs / 2, step=fs / N)
  plt.plot(freqsp, X_pl[:(N // 2)])
  # plotear frecuencias negativas
  freqsn = np.arange(-fs / 2, 0, step=fs / N)
  plt.plot(freqsn, X_pl[(N // 2):])
  # etiquetar eje x
  plt.xlabel("Frequencia (Hz)")
  plt.ylabel("Amplitud (log)")
  if len(xlim) >= 2:
    plt.xlim(xlim) 
  return X

def pasabanda(X,fs,finf,fsup):
  '''
    Filtro pasa banda ideal de las componentes en frecuencia X
    con frecuencia de muestreo fs, y frecuencias inferior y superior
    de la banda de paso finf y fsup.
  '''

  #inversion de muestras de frecuencias
  X2=np.fft.fftshift(X)
  
  #calculo de indices componentes de frecuencias
  nc=len(X)
  ifni =int ( -fsup*nc/(fs) + nc//2)
  ifns= int (-finf*nc/(fs) + nc//2)
  ifpi=int ( finf*nc/(fs) + nc//2)
  ifps =int ( fsup*nc/(fs) + nc//2)
  #eliminacion de componentes fuera de la banda de paso
  X2[:ifni]=0
  X2[ifns:ifpi]=0
  X2 [ifps:]=0
  #inversion de muestras de frecuencias
  X3=np.fft.ifftshift(X2)
  return X3

def xtriang(fs,periodo, tmax=1, tipo='t', polaridad='b'):
  '''
    señal triangular o diente de  sierra:
    primer argumento, frecuencia de muestreo
    arreglo de valores de la funcion en el tiempo
    segundo argumento, periodo del diente de sierra
    tercer argumento, tiempo de simulacion
    cuarto argumento, tipo: 't'riangular /\/\/\, 'c'reciente  /|/|/|, 'd'ecreciente  |\|\|\
    quinto argumento, polaridad: 'u'nipolar (0,1), 'b'ipolar (-1,1)
    entrega x, t: valores de la funcion, valores de tiempo 
  '''
  t = np.arange(0, tmax, step=1. / fs)
  if tipo=='c':
    x = np.mod(t,periodo)
  elif tipo=='d':
    x = periodo-np.mod(t,periodo)
  elif tipo=='t':
    x = 1-np.abs(periodo/2-np.mod(t,periodo))
  if polaridad=='u':
    x=normalizar(x)
  else:
    x=normalizarb(x)
  return x,t

def xpulsos(fs, periodo, ciclo, tmax=1, polaridad='b'):
  '''
    señal pulsos
    primer argumento, frecuencia de muestreo
    segundo argumento, periodo de la señal
    tercer argumento, ciclo de trabajo (entre 0 y 1)
    cuarto argumento, tiempo de simulacion
    quinto argumento, polaridad: 'u'nipolar (0,1), 'b'ipolar (-1,1)
    entrega x, t: valores de la funcion, valores de tiempo 
  '''
  t = np.arange(0, tmax, step=1. / fs)
  xx=t-t
  x = normalizar(np.mod(t,periodo))
  xx[x<ciclo]=1
  if polaridad=='u':
    xx=normalizar(xx)

def xcos(fs, f, tmax=1, a=1, ph=0):
  '''
    señal cosenoidal
    primer argumento, frecuencia de muestreo
    segundo argumento, frecuencia de la cosenoidal
    tercer argumento, tiempo de simulacion
    cuarto argumento, amplitud
    quinto argumento, fase en radianes
    entrega x, t: valores de la funcion, valores de tiempo 
  '''
  t = np.arange(0, tmax, step=1. / fs)
  x = a*np.cos(2*np.pi*f*t+ph)
  return x,t

def potencia(x):
  '''
    potencia de un vector de muestras,
    suma de los cuadrados de los elementos del vector x, dividido la longitud del vector
    valido para entradas reales y complejas, en dominio del tiempo y de la frecuencia
  '''
  p=np.sum(x*x.conjugate())/len(x)
  return p

def energia(x):
  '''
    energia de un vector de muestras,
    suma de los cuadrados de los elementos del vector x
    valido para entradas reales y complejas, en dominio del tiempo y de la frecuencia
  '''
  e=np.sum(x*x.conjugate())
  return e
  
def calc_espectrograma(x, fs, window_size, overlap):
  '''
    Crea un espectrograma a partir de un vector de muestras de tiempo.

    Argumentos:
    x: Array de muestras de tiempo.
    fs: Frecuencia de muestreo.
    window_size: Tamaño de la ventana de análisis (en muestras).
    overlap: Superposición entre ventanas consecutivas (como fracción, por ejemplo, 0.5 para una superposición del 50%).

    Retorna:
    espectrograma: Matriz 2D que representa el espectrograma.
    freqs: Array de frecuencias.
    times: Array de instantes de tiempo.
  '''
  # Calcula el número de muestras superpuestas
  hop_size = int(window_size * (1 - overlap))
  # Calcula el número de ventanas de tiempo
  num_windows = (len(x) - window_size) // hop_size + 1
  # Inicializa la matriz del espectrograma
  espectrograma = np.zeros((window_size, num_windows), dtype=complex)
  # Aplica la ventana y realiza la transformada de Fourier en cada ventana
  for i in range(num_windows):
      inicio = i * hop_size
      fin = inicio + window_size
      ventana = np.hanning(window_size)
      muestras_ventaneadas = x[inicio:fin] * ventana
      espectrograma[:, i] = np.fft.fft(muestras_ventaneadas)
  # Obtiene las frecuencias e instantes de tiempo
  freqs = np.fft.fftfreq(window_size, 1 / fs)
  times = np.arange(num_windows) * hop_size / fs
  return np.abs(espectrograma), freqs, times

def graf_espectrograma_3d(x, fs, window_size, overlap, azim=45, elev=-45):
  '''
    Grafica un espectrograma 3D de un vector de muestras de tiempo.

    Argumentos:
    x: Array de muestras de tiempo.
    fs: Frecuencia de muestreo.
    window_size: Tamaño de la ventana de análisis (en muestras).
    overlap: Superposición entre ventanas consecutivas (como fracción, por ejemplo, 0.5 para una superposición del 50%).
    azim: azimut de la camara
    elev: elevación de la camara  

    azim=0, elev=90, visto desde eje de frecuencias
    azim=90, elev=90, visto desde arriba, tiempo vertical
    azim=90, elev= 0, visto desde arriba, tiempo horizontal
    azim=0, elev=0, 
  '''
  # Crea el espectrograma
  espectrograma, freqs, times = calc_espectrograma(x, fs, window_size, overlap)
  # Crea una malla para la graficación del espectrograma
  times_mesh, freqs_mesh = np.meshgrid(times, freqs)
  # Grafica el espectrograma en 3D
  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(freqs_mesh,times_mesh, np.abs(espectrograma), cmap='jet')
  ax.set_ylabel('Tiempo (s)')
  ax.set_xlabel('Frecuencia (Hz)')
  ax.set_zlabel('Magnitud')
  ax.set_title('Espectrograma 3D')
  ax.view_init(azim, elev)
  plt.show()
  return np.abs(espectrograma), freqs, times
    
def graf_espectrograma(x, fs, window_size, overlap):
  '''
  Grafica el espectrograma en vista superior utilizando pcolor.
    Argumentos:
    x: Array de muestras de tiempo.
    fs: Frecuencia de muestreo.
    window_size: Tamaño de la ventana de análisis (en muestras).
    overlap: Superposición entre ventanas consecutivas (como fracción, por ejemplo, 0.5 para una superposición del 50%).
  '''
  # Crea el espectrograma
  espectrograma, freqs, times = calc_espectrograma(x, fs, window_size, overlap)
  # Grafica el espectrograma en vista superior utilizando pcolor
  plt.figure(figsize=(10, 6))
  plt.pcolor(times, freqs, np.abs(espectrograma), cmap='jet')
  plt.colorbar(label='Magnitud')
  plt.xlabel('Tiempo (s)')
  plt.ylabel('Frecuencia (Hz)')
  plt.title('Espectrograma en Vista Superior')
  plt.show()
  return np.abs(espectrograma), freqs, times

def generar_ofdm(bits, num_portadoras, cp_length):
  '''
    genera una señal ofdm
    primer argumento, bits a transmitir
    segundo argumento, numero de portadoras
    tercer argumento, longitud del prefijo cíclico
 '''
  # Parámetros de la señal OFDM
  num_bits = len(bits)
  num_subportadoras = num_portadoras - 1
  total_simbolos = num_bits // num_subportadoras
  # Conversión de bits a símbolos
  bits_por_simbolo = num_subportadoras
  bits_padded = np.append(bits, np.zeros(num_subportadoras - (num_bits % num_subportadoras)))
  simbolos = np.reshape(bits_padded, (total_simbolos, num_subportadoras))
  #simbolos = np.reshape(bits_padded, (total_simbolos, bits_por_simbolo))
  # Asignación de portadoras a los símbolos
  portadoras = np.zeros((total_simbolos, num_portadoras), dtype=complex)
  portadoras[:, 1:num_portadoras] = simbolos
  # Transformada de Fourier inversa (IFFT) para cada símbolo
  simbolos_ifft = np.fft.ifft(portadoras, axis=1)
  # Prefijo Cíclico (CP)
  simbolos_cp = np.concatenate((simbolos_ifft[:, -cp_length:], simbolos_ifft), axis=1)
  # Concatenación de los símbolos con CP
  s_ofdm = np.reshape(simbolos_cp, (-1,))
  return s_ofdm

def recibir_ofdm(s_ofdm, num_portadoras, cp_length):
  '''
    decodifica una señal ofdm
    primer argumento, señal ofdm
    segundo argumento, numero de portadoras
    tercer argumento, longitud del prefijo cíclico
  '''
  # Reorganización de la señal en bloques de símbolos con CP
  simbolos_cp = np.reshape(s_ofdm, (-1, num_portadoras + cp_length))
  # Remoción del prefijo cíclico (CP)
  simbolos = simbolos_cp[:, cp_length:]
  # Transformada de Fourier (FFT) para cada símbolo
  simbolos_fft = np.fft.fft(simbolos, axis=1)
  # Eliminación de la portadora nula y las portadoras negativas
  espectro = simbolos_fft[:, 1:(num_portadoras // 2)]
  # Conversión de los símbolos FFT a bits
  bits = np.round(np.real(espectro)).flatten()
  bits = bits.astype(int)
  return bits

def canal_inalambrico(senal, fs, snr_dB, delay_spread):
  '''
    genera un desvanecimiento multicamino y agrega ruido a una señal
    primer argumento, señal a transmitir por el canal
    segundo argumento, frecuencia de muestreo
    tercer argumento, relación señal a ruido en db
    cuarto argumento, dipersión de retardo
 '''
  # Generar desvanecimiento multicamino.
  h = np.sqrt(0.5) * (np.random.randn() + 1j * np.random.randn()) * np.exp(-1j * 2 * np.pi * np.random.rand())
  # Ajustar retardo en función de la frecuencia de muestreo
  delay_spread_samples = int(delay_spread * fs)
  # Aplicar desvanecimiento multicamino a la señal
  senal_desvanecida = np.convolve(senal, h)
  senal_desvanecida = senal_desvanecida[:len(senal)]  # Ajustar longitud de la señal desvanecida
  # Generar ruido gaussiano
  snr = 10 ** (snr_dB / 10)  # Conversión de dB a escala lineal
  var_ruido = np.var(senal_desvanecida) / snr
  ruido = np.sqrt(var_ruido / 2) * (np.random.randn(len(senal_desvanecida))+ 1j * np.random.randn(len(senal_desvanecida)))
  # Agregar ruido a la señal desvanecida
  senal_con_ruido = senal_desvanecida + ruido
  return senal_con_ruido
 
print("listo!")
