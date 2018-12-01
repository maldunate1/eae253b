import sys
import pandas as pd 
import os
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def listaProductosDistintos(bd):
	return bd.Item.unique()

def numeroProductosDistintos(bd):
	return len(listaProductosDistintos(bd))

def numeroRegistros(bd):
	return len(bd)

def numeroTransacciones(bd):
	return getLastRow(bd)['Transaction'].item()

def getLastRow(bd):
	return bd.iloc[-1:]

def agruparPorMes(df):
	df['YearMonth'] = pd.to_datetime(df['Date']).map(lambda dt: dt.replace(day=1))
	return df.groupby('YearMonth')['Transaction'].nunique()

def leerBD():
	dirname = os.path.dirname(__file__)
	filename = os.path.join(dirname, 'datosCanasta.csv')
	myFile = pd.read_csv(filename, sep=',')
	return myFile

def leerMatriz():
	dirname = os.path.dirname(__file__)
	filename = os.path.join(dirname, 'matriz.csv')
	myFile = pd.read_csv(filename, sep=',')
	return myFile

def productosMasVendidos(df):
	return df.Item.value_counts(sort=True)

def listaItems(df):
	return df.Item.unique().tolist()

def listaTransacciones(df):
	listaOfListas = []
	lista = []
	transaccionAnterior	= df.Transaction.iloc[0]
	for row in df.itertuples(index=True, name='Pandas'):
		if transaccionAnterior == getattr(row, "Transaction"):
			lista.append(getattr(row, "Item"))
		else:
			listaOfListas.insert(transaccionAnterior, lista)
	
		transaccionAnterior = getattr(row, "Transaction")
		pass
	return listaOfListas

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def transformToMatrix(bd):
	bd.drop(columns=['Date', 'Time'], inplace=True)
	bd['Quantity'] = 1
	# print(bd)
	basket = (bd
          .groupby(['Transaction', 'Item'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Transaction'))

	basket_sets = basket.applymap(encode_units)
	return basket_sets

def listaFrequentItemSets(bd):
	basket = transformToMatrix(bd)
	frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
	rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
	return rules

def support(X,Y,df):
	lx = frozenset(X)
	ly = frozenset(Y)
	print(df[(df.antecedents == lx) & (df.consequents == ly)].support)
	return 

def confidence(X,Y,df):
	lx = frozenset(X)
	ly = frozenset(Y)
	print(df[(df.antecedents == lx) & (df.consequents == ly)].confidence)
	return 

def lift(X,Y,df):
	lx = frozenset(X)
	ly = frozenset(Y)
	print(df[(df.antecedents == lx) & (df.consequents == ly)].lift)
	return 

def conviction(X,Y,df):
	lx = frozenset(X)
	ly = frozenset(Y)
	print(df[(df.antecedents == lx) & (df.consequents == ly)].conviction)
	return 

def pregunta1(bd):
	numeroTransacciones(bd)
	print('\n\n RESPUESTAS DEL PREGUNTA 1 \n')
	#Pregunta 1.1
	print("Numero de registros de la bd: "+ str(numeroRegistros(bd)) + '\n')	
	#Pregunta 1.2
	print("Numero de transacciones de la bd: "+ str(numeroTransacciones(bd))+ '\n')	
	#Pregunta 1.3
	print("Numero de productos distintos de la bd: " + str(numeroProductosDistintos(bd)) + '\n')	
	#Pregunta 1.4
	print("Transacciones por mes en la base de datos: " + '\n')
	print(agruparPorMes(bd))
	#Pregunta 1.5
	print('\n' + "Lista de productos ordenada por cantidad de ventas: " + '\n')	
	print(productosMasVendidos(bd))

def pregunta2(bd):
	tabla = listaFrequentItemSets(bd)
	
	#ESTOS SON LAS LISTAS QUE HAY QUE CAMBIAR PARA VER COMO SE COMPORTAN LAS FUNCIONES, OJO QUE SI NO APARECE NADA ES PORQUE NO EXISTE ESA COMBINACION
	
	X = ['Coffee']
	Y = ['Cake']
	
	print('\n Support de las reglas es: ')
	support(X,Y,tabla)
	print('\n confidence de las reglas es: ')
	confidence(X,Y,tabla)
	print('\n Support de las reglas es: ')
	lift(X,Y,tabla)
	print('\n Support de las reglas es: ')
	conviction(X,Y,tabla)

	return

def bonus(bd):
	tabla_association_rules = listaFrequentItemSets(bd)
	print('\n\n RESPUESTAS DEL BONUS \n')
	# Bonus 3.1
	print('\n' + "5 reglas con mayor support: (Se muestran 10 porque es igual X->Y que Y->X)" + '\n')
	print(tabla_association_rules.nlargest(10, 'support'))
	# Bonus 3.2
	print('\n' + "5 reglas con mayor confidence: " + '\n')
	print(tabla_association_rules.nlargest(5, 'confidence'))
	# Bonus 3.3
	print('\n' + "5 reglas con mayor lift: (Se muestran 10 porque es igual X->Y que Y->X)" + '\n')
	print(tabla_association_rules.nlargest(10, 'lift'))
	# Bonus 3.4
	print('\n' + "5 reglas con mayor conviction: " + '\n')
	print(tabla_association_rules.nlargest(5, 'conviction'))
	# Bonus 3.5
	print("\nElijo la regla (Coffee)->(Cake) que es la con mayor support (5,4%) esto quiere decir que es la regla de asociacion que mas ocurre. Tiene un confidence de 11,4%% lo que significa que casi 1 de cada 10 veces estos productos se compran juntos. Tambien un lift de 1,1; lo cual es alto, lo que implica que existe una gran asociacion entre estos productos. Y un conviction de 1, este valor es alto, por lo que (Cake) depende fuertemente de (Coffee)") 

if __name__ == "__main__":
	bd = leerBD()
	
	# RESPUESTAS A LA PREGUNTA 1
	pregunta1(bd)

	bd = leerBD()
	# RESPUESTAS A LA PREGUNTA 2
	pregunta2(bd)
	
	bd = leerBD()
	# BONUS
	bonus(bd)


	possible_products = ["Afternoon with the baker","Alfajores","Argentina Night","Art Tray","Bacon","Baguette","Bakewell","Bare Popcorn","Basket","Bowl Nic Pitt","Bread","Bread Pudding","Brioche and salami","Brownie","Cake","Caramel bites","Cherry me Dried fruit","Chicken Stew","Chicken sand","Chimichurri Oil","Chocolates","Christmas common","Coffee","Coffee granules ","Coke","Cookies","Crepes","Crisps","Drinking chocolate spoons ","Duck egg","Dulce de Leche","Eggs", "Ella\'s Kitchen Pouches","Empanadas","Extra Salami or Feta","Fairy Doors","Farm House","Focaccia","Frittata","Fudge","Gift voucher","Gingerbread syrup","Granola","Hack the stack","Half slice Monster ","Hearty & Seasonal","Honey","Hot chocolate","Jam","Jammie Dodgers","Juice","Keeping It Local","Kids biscuit","Lemon and coconut","Medialuna","Mighty Protein","Mineral water","Mortimer","Muesli","Muffin","My-5 Fruit Shoot","NONE","Nomad bag","Olum & polenta","Panatone","Pastry","Pick and Mix Bowls","Pintxos","Polenta","Postcard","Raspberry shortbread sandwich","Raw bars","Salad","Sandwich","Scandinavian","Scone","Siblings","Smoothies","Soup","Spanish Brunch","Spread","Tacos/Fajita","Tartine","Tea","The BART","The Nomad","Tiffin","Toast","Truffles","Tshirt","Valentine\'s card","Vegan Feast","Vegan mincepie","Victorian Sponge"]

# Afternoon with the baker	Alfajores	Argentina Night	Art Tray	Bacon	Baguette	Bakewell	Bare Popcorn	Basket	Bowl Nic Pitt	Bread	Bread Pudding	Brioche and salami	Brownie	Cake	Caramel bites	Cherry me Dried fruit	Chicken sand	Chicken Stew	Chimichurri Oil	Chocolates	Christmas common	Coffee	Coffee granules 	Coke	Cookies	Crepes	Crisps	Drinking chocolate spoons 	Duck egg	Dulce de Leche	Eggs	Ella's Kitchen Pouches	Empanadas	Extra Salami or Feta	Fairy Doors	Farm House	Focaccia	Frittata	Fudge	Gift voucher	Gingerbread syrup	Granola	Hack the stack	Half slice Monster 	Hearty & Seasonal	Honey	Hot chocolate	Jam	Jammie Dodgers	Juice	Keeping It Local	Kids biscuit	Lemon and coconut	Medialuna	Mighty Protein	Mineral water	Mortimer	Muesli	Muffin	My-5 Fruit Shoot	Nomad bag	NONE	Olum & polenta	Panatone	Pastry	Pick and Mix Bowls	Pintxos	Polenta	Postcard	Raspberry shortbread sandwich	Raw bars	Salad	Sandwich	Scandinavian	Scone	Siblings	Smoothies	Soup	Spanish Brunch	Spread	Tacos/Fajita	Tartine	Tea	The BART	The Nomad	Tiffin	Toast	Truffles	Tshirt	Valentine's card	Vegan Feast	Vegan mincepie	Victorian Sponge
