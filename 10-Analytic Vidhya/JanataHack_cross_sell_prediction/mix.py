import pandas as pd 
l = pd.read_csv("lgbm80.csv")
c = pd.read_csv("cat78.csv")
x = pd.read_csv("xg80.csv")
t1 = pd.read_csv("t180.csv")
t2 = pd.read_csv("t280.csv")
nar = pd.read_csv("8583nar.csv")




l = l.sort_values(['id']).reset_index(drop=True)
c = c.sort_values(["id"]).reset_index(drop=True)
x = x.sort_values(["id"]).reset_index(drop=True)
t1 = t1.sort_values(["id"]).reset_index(drop=True)
t2 = t2.sort_values(["id"]).reset_index(drop=True)
nar = nar.sort_values(["id"]).reset_index(drop=True)


c1 = []
c2 = []
for i in  range(len(l)):
	ID = l["id"][i]

	ll = l["Response"][i]
	cc = c["Response"][i]
	xx = x["Response"][i]
	tt1 = t1["Response"][i]
	tt2 = t2["Response"][i]
	narr = nar["Response"][i]

	# avg = (ll*0.4) + (xx*0.4) + (cc*0.2)
	# avg = (ll*0.22) + (xx*0.22) + (tt1*0.22) + (tt2*0.22) + (cc*0.12)              #0.858536
	# avg = (ll*0.25) + (xx*0.25) + (tt1*0.25) + (tt2*0.25)                          #0.858519
	# avg = (ll*0.2) + (xx*0.2) + (tt1*0.2) + (tt2*0.2) + (cc*0.2)                   #0.858530


	# avg = (narr*0.28) + (ll*0.18) + (xx*0.18) + (tt1*0.18) + (tt2*0.18)                       #0.858901
	# avg = (narr*0.28) + (ll*0.15) + (xx*0.15) + (tt1*0.15) + (tt2*0.15)  + (cc*0.12)          #0.858915
	avg = (narr*0.28) + (ll*0.16) + (xx*0.16) + (tt1*0.16) + (tt2*0.16)  + (cc*0.08)          #0.858917      

	c1.append(ID)
	c2.append(avg)


diction = {"id":c1,"Response":c2}
filename = "blend3.csv"
df_fin = pd.DataFrame(diction)

df_fin.to_csv(filename,index=False)
