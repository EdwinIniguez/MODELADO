import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl, os
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns 

def convertir_columnas_fecha(df, columnas, normalizar=True):
    """
    Convierte múltiples columnas con fechas tipo string (con o sin separadores decimales raros) al formato datetime.
    Opcionalmente normaliza a nivel de fecha (elimina hora y nanosegundos).

    Parámetros:
    - df: DataFrame original.
    - columnas: lista de nombres de columnas a convertir.
    - normalizar: booleano. Si True, aplica .dt.normalize() para dejar solo la fecha.

    Retorna:
    - DataFrame con las columnas especificadas convertidas a datetime (y normalizadas si se desea).
    """
    for col in columnas:
        df[col] = pd.to_datetime(
            df[col].astype(str).str.replace(',', '.', regex=False),
            format='%Y-%m-%d %H:%M:%S.%f',
            errors='coerce'
        )
        if normalizar:
            df[col] = df[col].dt.normalize()
    return df

def calcular_monto_pagado_hasta_vencimiento(data):
    """
    Calcula el monto acumulado pagado por cada estudiante hasta la fecha de vencimiento de cada cuota.

    Parámetros:
        data (DataFrame): Debe contener las columnas 'CodigoMatricula', 'Fechapago', 'MontoPagado' y 'FechaVencimiento'.

    Retorna:
        DataFrame: Devuelve el mismo DataFrame con una columna nueva 'MontoPagadoAcumFV'.
    """

    # 1. Pagos válidos (con fecha de pago)
    pagos = data[['CodigoMatricula', 'Fechapago', 'MontoPagado']].dropna(subset=['Fechapago']).copy()

    # 2. Cuotas con índice para asignar después
    cuotas = data[['CodigoMatricula', 'FechaVencimiento']].copy()
    cuotas['id_row'] = cuotas.index  # para mapear luego al orden original

    # 3. Merge entre cuotas y pagos
    merged = cuotas.merge(pagos, on='CodigoMatricula', how='left')

    # 4. Filtrar pagos hasta la fecha de vencimiento
    merged = merged[merged['Fechapago'] <= merged['FechaVencimiento']]

    # 5. Agrupar y sumar pagos válidos por fila original
    acumulado = merged.groupby('id_row')['MontoPagado'].sum()

    # 6. Asignar al dataframe original
    data['MontoPagadoAcumFV'] = data.index.to_series().map(acumulado).fillna(0)

    return data

def cluster_interaccion(df):
    """
    Calcula la variable 'NivelInteraccion' agrupando estudiantes por KMeans
    a partir de variables estándar de interacción.
    
    Parámetros:
        df: DataFrame original (debe contener las columnas necesarias)

    Retorna:
        df con columna 'NivelInteraccion' agregada y variables originales eliminadas
    """
    columnas_interaccion = ["TotalCompromisosCuotaAnterior", "CCorreosEnviados", "CWhatsAppEnviados", "MinutosTotales"]

    # Verificar que todas las columnas existan
    if not all(col in df.columns for col in columnas_interaccion):
        raise ValueError("Faltan una o más columnas requeridas para la clusterización de interacción.")
    
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[columnas_interaccion] = scaler.fit_transform(df_scaled[columnas_interaccion])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_scaled["NivelInteraccion"] = kmeans.fit_predict(df_scaled[columnas_interaccion])

    return df_scaled.drop(columns=columnas_interaccion)

def calcular_cuotas_impagas(data1):
    """
    Calcula la cantidad de cuotas impagas acumuladas por cada estudiante (CodigoMatricula),
    considerando que una cuota se considera impaga si:
    - No tiene fecha de pago (NaN)
    - O fue pagada después de la fecha de vencimiento de la cuota actual
    
    Parameters:
        data (DataFrame): DataFrame con columnas ['CodigoMatricula', 'NroCuota', 'FechaVencimiento', 'Fechapago']
    
    Returns:
        DataFrame: Mismo DataFrame de entrada con una nueva columna 'CuotasImpagasPrevias'
    """
    
    # Asegurar orden por estudiante y número de cuota
    df = data1.sort_values(by=['CodigoMatricula', 'NroCuota']).reset_index(drop=True)

    # Función interna para contar impagas dentro del grupo
    def contar_impagas(grupo):
        resultado = []
        for i, fila in grupo.iterrows():
            fecha_ref = fila['FechaVencimiento']
            nrocuota_actual = fila['NroCuota']

            # Filtrar cuotas hasta la actual
            cuotas_previas = grupo[grupo['NroCuota'] <= nrocuota_actual]

            # Considerar como impaga si no tiene fecha o pagó después del vencimiento actual
            cuotas_impagas = cuotas_previas[
                (cuotas_previas['Fechapago'].isna()) |
                (cuotas_previas['Fechapago'] > fecha_ref)
            ]

            resultado.append(len(cuotas_impagas))
        grupo['CuotasImpagasPrevias'] = resultado
        return grupo

    # Aplicar por estudiante
    return df.groupby('CodigoMatricula', group_keys=False).apply(contar_impagas)

def agregar_suma_temporal(df_info, df_Final, columna_suma, nombre_columna_resultado, columna_fecha_evento,nombre_columna_DataFrame):
    """
    Agrupa y suma una columna basada en eventos ocurridos dentro de un rango mensual,
    y une el resultado al df_Final.

    Parámetros:
    - df_merge: DataFrame que contiene eventos por fecha
    - df_Final: DataFrame donde se consolidará la suma por estudiante y mes
    - columna_suma: Nombre de la columna que se va a sumar
    - nombre_columna_resultado: Nombre de la columna nueva con el resultado
    - columna_fecha_evento: Nombre de la columna de fecha a evaluar (ej. 'FechaVisualizacion', 'FechaPago', etc.)
    - df_info: es un df_visual
    - nombre_columna_DataFrame: El nombre de la columna para el DF Final
    Retorna:
    - df_Final con la columna de suma mensual agregada
    """

    # 1. Filtrar eventos con fecha válida
    df_info = df_info[df_info[columna_fecha_evento].notna()]

    # 2. Unir con el calendario de meses
    df_merge = df_Final[['CodigoMatricula', 'FechaMatricula', 'FechaInicioMes', 'FechaFinMes']].merge(
        df_info, on='CodigoMatricula', how='left'
    )

    # 3. Calcular condición de inclusión según coincidencia con FechaMatricula
    df_merge['FechaInicioFiltrada'] = df_merge.apply(
        lambda row: row['FechaInicioMes'] - pd.Timedelta(days=30) if row['FechaMatricula'].date() == row['FechaInicioMes'].date()
        else row['FechaInicioMes'], axis=1)

    # 4. Filtrar eventos dentro del rango válido
    df_filtrado = df_merge[
        (df_merge[columna_fecha_evento] > df_merge['FechaInicioFiltrada']) &
        (df_merge[columna_fecha_evento] <= df_merge['FechaFinMes'])
    ]

    # 5. Sumar duración por mes
    suma_duracion = df_filtrado.groupby(['CodigoMatricula', 'FechaInicioMes', 'FechaFinMes'])[columna_suma]\
        .sum().reset_index(name=nombre_columna_resultado)

    # 6. Unir con df_Final
    df_Final = df_Final.merge(suma_duracion, on=['CodigoMatricula', 'FechaInicioMes', 'FechaFinMes'], how='left')

    # 7. Rellenar donde no hubo visualización
    df_Final[nombre_columna_resultado] = df_Final[nombre_columna_resultado].fillna(0)

    # 8. Calcular acumulado del avance por estudiante
    df_Final[nombre_columna_DataFrame] = df_Final.groupby('CodigoMatricula')[nombre_columna_resultado].cumsum()

    return df_Final


def calcular_dias_ultima_llamada_efectiva(df_cuotas, df_llamadas):
    # Filtrar solo llamadas efectivas
    df_llamadas_efectivas = df_llamadas[df_llamadas['LlamadaEfectiva'] == 1]

    # Cruzar cuotas con llamadas por CodigoMatricula
    df_merge = df_cuotas.merge(df_llamadas_efectivas, on='CodigoMatricula', how='left')

    # Quedarse solo con llamadas anteriores o iguales a la fecha de vencimiento
    df_validas = df_merge[df_merge['FechaActividad'] <= df_merge['FechaVencimiento']].copy()

    # Calcular la diferencia de días
    df_validas['DiasDesdeLlamada'] = (df_validas['FechaVencimiento'] - df_validas['FechaActividad']).dt.days

    # Obtener la llamada más reciente (menor cantidad de días desde la fecha)
    idx = df_validas.groupby(['CodigoMatricula', 'FechaVencimiento'])['DiasDesdeLlamada'].idxmin()

    # Extraer resultados y renombrar
    df_resultado = df_validas.loc[idx, ['CodigoMatricula', 'FechaVencimiento', 'DiasDesdeLlamada']]
    df_resultado = df_resultado.rename(columns={'DiasDesdeLlamada': 'DiasUltimaLlamadaEfectiva'})

    # Volver a unir con las cuotas originales
    df_final = df_cuotas.merge(df_resultado, on=['CodigoMatricula', 'FechaVencimiento'], how='left')

    return df_final

def calcular_duracion(row):
    """
    Calcula la duración en meses entre la fecha de matrícula y la fecha de finalización del programa,
    considerando un ajuste si el día del mes final es mayor que el del inicio.

    Parámetros:
    -----------
    row : pd.Series
        Fila de un DataFrame que contiene las columnas 'FechaMatricula' y 'FechaFinalizacionPrograma',
        ambas de tipo datetime.

    Retorna: Duración total del programa en meses.
    """
    
    fecha_ini = row['FechaMatricula']
    fecha_fin = row['FechaFinPrograma']
    
    # Calcular la diferencia base en meses
    meses = (fecha_fin.year - fecha_ini.year) * 12 + (fecha_fin.month - fecha_ini.month)
    
    # Verificar si los días de fin superan a los de inicio => agregar un mes más
    if fecha_fin.day > fecha_ini.day:
        meses += 1
    meses+2
    return meses    

def agrupar_variables_categoricas(df):
  """Transformacion de las variables categoricas a numericas, segun su importancia dada por la regresion lineal
  """  
  dic_Industria={1: ['Pesca',
    'Comercio Exterior',
    'Defensa',
    'Química',
    'ONGS',
    'Siderurgia',
    'Correo/Mensajería',
    'Gastronomia',
    'Editorial/Medios',
    'Ganaderia',
    'Seguros',
    'Farmaceutica',
    'Petroquimica',
    'Publicidad / Marketing / RRPP',
    'Informatica/Tecnologia/Internet',
    'Metal-Mecanica'],
  2: ['Banca/Financiera',
    'Alimenticia',
    'Telecomunicaciones',
    'Retail/ Supermercado / Consumo Masivo',
    'Gobierno',
    'Servicios',
    'Logistica/Transporte',
    'Transporte Aéreo',
    'Arquitectura/Construccion/Inmobiliaria',
    'Hoteleria',
    'Forestal',
    'Imprenta',
    'Entretenimiento',
    'Salud',
    'Automotriz'],
  3: ['Textil',
    'Petroleo / Gas',
    'Consultoria',
    'Jurídica',
    'Educación',
    'Energia',
    'Artesanal',
    'Sin industria',
    'Biotecnologia',
    'Manufactura',
    'Tabacalera',
    'Agro-Industrial',
    'Papelera',
    'Otra',
    'Higiene y Perfumería',
    'Mineria']}
  dic_Cargo={1: ['Instrumentista',
    'Gerente Adjunto',
    'Prevencionista',
    'Formador y Capacitador',
    'Gerente General/ Apoderado',
    'Director/Presidente/CEO',
    'Atención al cliente',
    'Asesor Comercial',
    'Ejecutivo',
    'Practicante'],
  2: ['Socio/Accionista',
    'Analista',
    'Operario',
    'Investigador',
    'Auditor',
    'Asistente',
    'Asistente/Analista',
    'Jefe/Encargado',
    'Otros',
    'Funcionario Publico'],
  3: ['Encargado',
    'Consultor',
    'Coordinador',
    'Superintendente',
    'Docente',
    'Inspector',
    'Gerente de Área/Unidad',
    'Jefe',
    'Estudiante',
    'Supervisor']}
  dic_Trabajo={1: ['Tecnologia Sistemas y Telecomunicaciones',
    'Comercial y/o Ventas',
    'Seguridad de la Información',
    'Investigación',
    'Analytics Big Data y/o Business Intelligence',
    'Recursos Humanos y/o Entrenamiento',
    'Secretaria y Recepcion',
    'Aduana y Comercio Exterior',
    'Legal'],
  2: ['Oficios y Otros',
    'Comunicacion Relaciones Institucionales y Publicas',
    'Gerencia y Direccion General',
    'Produccion y/o Manufactura',
    'Marketing y Publicidad',
    'Seguridad Ocupacional y/o Industrial',
    'Abastecimiento y Logistica',
    'Proyectos'],
  3: ['Cobranza',
    'Mantenimiento',
    'Sin Area de trabajo',
    'Otros',
    'Calidad',
    'Operaciones',
    'Ingenieria',
    'Administracion Contabilidad y Finanzas']}
  dic_Formacion={1: ['Ing. Sonido',
    'Relaciones Industriales',
    'Ing. Pesquera',
    'Ing. Hidraulica',
    'Turismo',
    'Sociología',
    'Biotecnologia',
    'Ing. Aerospacial',
    'Quimica',
    'Ing. Forestal',
    'Literatura',
    'Ing. en Materiales',
    'Topografia',
    'Odontologia y Afines',
    'Diseño de Modas',
    'Ing. Estadistica',
    'Diseño Industrial',
    'Derecho',
    'Cobranza',
    'Informatica y Afines',
    'Ing. Comercial',
    'Relaciones Publicas',
    'Diseño Grafico',
    'Adm. Empresas',
    'Terapeuta Físico',
    'Construccion',
    'Programacion y Afines',
    'Contabilidad',
    'Hidraulica',
    'Arqueologo',
    'Comercio Exterior'],
  2: ['Gastronomia/cocina',
    'Matemática',
    'Estadística',
    'Tecnologia Médica/Laboratorio',
    'Tecnologia de la Informacion',
    'Geofisica',
    'Mantenimiento y Afines',
    'Veterinaria',
    'Ing. Naval',
    'Agro-Negocios',
    'Ing. Quimica',
    'Tecnologia de Alimentos',
    'Produccion y Afines',
    'Economia',
    'Ing. Civil',
    'Metal Mecanica y Afines',
    'Nutricion',
    'Ing. Informatica',
    'Electronica',
    'Ing. Alimentos',
    'Geografia',
    'Periodismo',
    'Ing. Geologica',
    'Ing. Electronica',
    'Geologia',
    'Sin Area de formacion',
    'Ing. Telecomunicaciones',
    'Mecanica y Afines',
    'Ing. en Sistemas',
    'Farmacia'],
  3: ['Ing. Sanitaria',
    'Biologia',
    'Enfermeria',
    'Otros',
    'Secretariado y Afines',
    'Ing. Ambiental',
    'Electricidad',
    'Militar',
    'Relaciones Internac.',
    'Ing. Mecanica',
    'Marketing y Afines',
    'Telecomunicaciones',
    'Trabajo Social',
    'Ing. Petroleo',
    'Educacion',
    'Computacion y Afines',
    'Ing. en Minas',
    'Ing. Agronomo-Agropecuario',
    'Ing. Metalurgica',
    'Ciencias Físicas',
    'Ing. Recursos Hidricos',
    'Ing. Industrial',
    'Arquitectura',
    'Ing. Electrica',
    'Medicina y Afines',
    'Finanzas y Afines',
    'Comunicacion y Afines',
    'Hoteleria y Afines',
    'Negocios y Afines',
    'Psicologia']}
  VersionMatricula_dict={'Gerencial':4, 'Sin Version':1, 'Profesional':3, 'Basica':2}

  def agrupado_pais(valor):
      paises={'Perú':1,'Mexico':2,'Colombia':3,'Bolivia':4}
      if valor in paises.keys():
          pais=paises[valor]
          return pais
      return 5

  diccionarios_categoricos=[dic_Industria,dic_Cargo, dic_Trabajo,dic_Formacion]

  # Invertimos y reasignamos
  for i in range(len(diccionarios_categoricos)):
      diccionarios_categoricos[i] = {
          cat: seg for seg, cats in diccionarios_categoricos[i].items() for cat in cats}
  dic_Industria,dic_Cargo, dic_Trabajo,dic_Formacion = diccionarios_categoricos

  diccionario={'Industria':'Sin industria', 'Cargo':'Otros','AreaTrabajo':'Sin Area de trabajo',
  'AreaFormacion':'Sin Area de formacion'}
  for i in list(diccionario.keys()):
    df.loc[:,i]=df[i].fillna(diccionario[i])

  le = LabelEncoder()
  df.loc[:,'AreaProgramaGroup']=le.fit_transform(df['AreaPrograma'])
  df.loc[:,'AreaTrabajoGroup']=df['AreaTrabajo'].map(dic_Trabajo)
  df.loc[:,'IndustriaGroup']=df['Industria'].map(dic_Industria)
  df.loc[:,'CargoGroup']=df['Cargo'].map(dic_Cargo)
  df.loc[:,'AreaFormacionGroup']=df['AreaFormacion'].map(dic_Formacion)
  df.loc[:,'PaisGroup']=df['NombrePais'].apply(agrupado_pais)
  df.loc[:,'VersionProgramaGroup']=df['VersionPrograma'].map(VersionMatricula_dict)
  df=df.drop(columns=['Industria', 'Cargo','AreaTrabajo', 'AreaFormacion', 
               'NombrePais','VersionPrograma','AreaPrograma'])
  return df

def cluster_interaccion(df):
    """
    Calcula la variable 'NivelInteraccion' agrupando estudiantes por KMeans
    a partir de variables estándar de interacción.
    
    Parámetros:
        df: DataFrame original (debe contener las columnas necesarias)

    Retorna:
        df con columna 'NivelInteraccion' agregada y variables originales eliminadas
    """
    columnas_interaccion = ["TotalCompromisosCuotaAnterior", "CCorreosEnviados", "CWhatsAppEnviados", 'MinutosTotales']
    
    # Verificar que todas las columnas existan
    if not all(col in df.columns for col in columnas_interaccion):
        raise ValueError("Faltan una o más columnas requeridas para la clusterización de interacción.")
    
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[columnas_interaccion] = scaler.fit_transform(df_scaled[columnas_interaccion])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_scaled["NivelInteraccion"] = kmeans.fit_predict(df_scaled[columnas_interaccion])
    
    return df_scaled.drop(columns=columnas_interaccion)

def cluster_categoricas(df):
    """
    Calcula la variable 'NivelInteraccion' agrupando estudiantes por KMeans
    a partir de variables estándar de interacción.
    
    Parámetros:
        df: DataFrame original (debe contener las columnas necesarias)

    Retorna:
        df con columna 'NivelInteraccion' agregada y variables originales eliminadas
    """
    columnas_interaccion = ['AreaProgramaGroup', 'AreaTrabajoGroup', 'IndustriaGroup', 'CargoGroup',
       'AreaFormacionGroup', 'PaisGroup', 'VersionProgramaGroup']
    
    # Verificar que todas las columnas existan
    if not all(col in df.columns for col in columnas_interaccion):
        raise ValueError("Faltan una o más columnas requeridas para la clusterización de interacción.")
    
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[columnas_interaccion] = scaler.fit_transform(df_scaled[columnas_interaccion])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_scaled["NivelCategoricas"] = kmeans.fit_predict(df_scaled[columnas_interaccion])
    
    return df_scaled.drop(columns=columnas_interaccion)

def codificar_categorias(df, columna_categorica, columna_objetivo):
    """
    Retorna un diccionario de la categoria con los diferentes niveles"""

    diccionario={'Industria':'Sin industria', 'Cargo':'Otros','AreaTrabajo':'Sin Area de trabajo',
    'AreaFormacion':'Sin Area de formacion'}

    df[columna_categorica]=df[columna_categorica].fillna(diccionario[columna_categorica])
    
    # 1. Crear dummies
    dummies = pd.get_dummies(df[columna_categorica], drop_first=False)
    X = dummies.astype(int)
    y = df[columna_objetivo]

    
    # 2. Ajustar modelo de regresión logística
    modelo = LogisticRegression(solver='lbfgs',max_iter=1000)
    modelo.fit(X, y)

    # 3. Obtener coeficientes
    coeficientes = modelo.coef_[0]
    categorias = dummies.columns
    df_coef = pd.DataFrame({
        'categoria': categorias,
        'coeficiente': coeficientes,
        'coeficientes_abs':abs(coeficientes)
    }).sort_values('coeficientes_abs', ascending=False)

    # 4. Segmentar en 3 grupos (alto, medio, bajo impacto)
    def segmentar(df_coef):
        terciles = pd.qcut(df_coef['coeficientes_abs'], 3, labels=[3, 2, 1])  # 1 = más alto, 3 = más bajo
        df_coef['segmento'] = terciles.astype(int)
        segmentos = df_coef.groupby('segmento')['categoria'].apply(list).to_dict()
        return segmentos

    # # 5. Ejecutar segmentación y devolver resultado
    return segmentar(df_coef)

##Analisis EDA
def chi2_test_categoricas(df, target_col, alpha=0.05):
    resultados = []

    # Filtrar solo columnas categóricas (excluyendo la target)
    cat_cols = df.drop(columns=target_col).select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        # Crear tabla de contingencia
        tabla = pd.crosstab(df[col], df[target_col])

        # Aplicar test de Chi-cuadrado
        chi2, p, dof, expected = chi2_contingency(tabla)

        resultados.append({
            'Variable': col,
            'Chi2': chi2,
            'p-valor': p,
            'Significativo': 'Sí' if p < alpha else 'No'})
    return pd.DataFrame(resultados).sort_values('p-valor')

def plot_heatmap_correlacion(df, titulo='Matriz de Correlación entre Variables Numéricas', tamaño=(10, 8)):
    """
    Grafica un heatmap de correlación entre las variables numéricas de un DataFrame.
    
    Parámetros:
    - df: pandas.DataFrame
    - titulo: str, título del gráfico
    - tamaño: tuple, tamaño de la figura (ancho, alto)
    """
    # Seleccionar solo columnas numéricas
    numericas = df.select_dtypes(include=['float', 'int'])

    # Calcular la matriz de correlación
    matriz_corr = numericas.corr()

    # Crear el heatmap
    plt.figure(figsize=tamaño)
    sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

def plot_heatmap_correlacionObj(df, columna_objetivo,titulo='Matriz de Correlación entre Variables Numéricas',tamaño=(10, 8)):
    """
    Grafica un heatmap de correlación entre las variables numéricas de un DataFrame.

    Parámetros:
    - df: pandas.DataFrame
    - titulo: str, título del gráfico
    - tamaño: tuple, tamaño de la figura (ancho, alto)
    """
    # Seleccionar solo columnas numéricas
    numericas = df.drop(columns=columna_objetivo).select_dtypes(include=['float', 'int'])

    # Calcular la matriz de correlación
    matriz_corr = numericas.corr(method='spearman')

    # Crear el heatmap
    plt.figure(figsize=tamaño)
    sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()
