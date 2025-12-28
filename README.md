# Softmax klasifikator treniran mini-batch SGD algoritmom

## 1. Uvod

Cilj ovog projekta je implementacija **multiklasa softmax klasifikatora**
(multinomialne logističke regresije) i njegova obuka korišćenjem
**stohastičkog gradijentnog spusta sa mini-šaržama (mini-batch SGD)**.

Implementacija je realizovana **bez korišćenja gotovih softmax funkcija
ili optimizacionih algoritama**, u skladu sa zadatim ograničenjima.
Kao referenca za proveru korektnosti implementacije koristi se
gotova implementacija softmax klasifikatora iz biblioteke *scikit-learn*.

Sav eksperimentalni kôd nalazi se u fajlu `main.py`.

---

## 2. Skup podataka

Podaci se učitavaju iz datoteke: `res/multiclass_data.csv`

Struktura podataka je sledeća:

- poslednja kolona predstavlja **oznaku klase** (celobrojna vrednost),
- sve prethodne kolone predstavljaju **numeričke prediktore**.

U ovom radu koriste se **isključivo originalni prediktori**, bez
polinomijalne ekspanzije ili dodatnih transformacija.

---

## 3. Model

Korišćen je **softmax klasifikator**, koji za ulazni vektor prediktora
$\mathbf{x} \in \mathbb{R}^d$ računa verovatnoće klasa kao:

$$
p(y = k \mid \mathbf{x}) =
\frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}
{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
$$

gde je:

- $K$ broj klasa,
- $\mathbf{w}_k$ vektor težina za klasu $k$,
- $b_k$ pristrasni član.

---

## 4. Funkcija gubitka

Kao funkcija gubitka koristi se **negativna log-verodostojnost**
(cross-entropy gubitak):

$$
\mathcal{L} =
-\frac{1}{N} \sum_{i=1}^{N}
\log p(y_i \mid \mathbf{x}_i)
$$

gde je $N$ broj primera u obučavajućem skupu.

---

## 5. Optimizacija

Za optimizaciju parametara koristi se
**stohastički gradijentni spust sa mini-šaržama**.

Osnovne karakteristike obučavanja:

- podaci se **nasumično mešaju na početku svake epohe**,
- obuka se vrši u mini-šaržama veličine $m_{mb}$,
- jedan prolazak kroz ceo obučavajući skup predstavlja **jednu epohu**,
- nakon svake epohe meri se:
    - prosečan gubitak na trening skupu,
    - tačnost na validacionom skupu.

Implementacija algoritma nalazi se u funkciji `train` u fajlu `main.py`.

---

## 6. Eksperimentalna podešavanja

Eksperimenti su sprovedeni promenom sledećih hiper-parametara:

- stopa učenja $\alpha$,
- veličina mini-šarže $m_{mb}$.

Za svaku kombinaciju hiper-parametara beleženi su:

- tok konvergencije gubitka po epohama,
- tačnost na validacionom skupu,
- prosečno trajanje jedne epohe obučavanja.

Vreme epohe meri se korišćenjem funkcije `time()` iz standardnog Python
modula `time`.

---

## 7. Rezultati

Za analizu konvergencije generisani su sledeći grafici:

- gubitak na trening skupu u funkciji epohe,
- tačnost na validacionom skupu u funkciji epohe.

Prikazano je ukupno pet konfiguracija:

1. optimalna kombinacija $(\alpha^\*, m_{mb}^\*)$,
2. $\alpha^\*$ i prevelika mini-šarža,
3. $\alpha^\*$ i premala mini-šarža,
4. $m_{mb}^\*$ i prevelika stopa učenja,
5. $m_{mb}^\*$ i premala stopa učenja.

Ukupno vreme do konvergencije procenjeno je kao proizvod:

$$
T_{uk} = T_{epohe} \cdot N_{epoha}
$$

gde je $N_{epoha}$ broj epoha potreban da se dostigne vrednost gubitka
bliska konačnoj.

---

## 8. Validacija implementacije

Radi provere korektnosti ručne implementacije, rezultati su upoređeni sa
referentnim softmax klasifikatorom iz biblioteke *scikit-learn*,
implementiranim u fajlu `check.py`.

Dobijene tačnosti na trening i validacionom skupu su uporedive, što
potvrđuje ispravnost implementiranog algoritma.

---

## 9. Zaključak

U ovom radu uspešno je implementiran softmax klasifikator treniran
mini-batch SGD algoritmom, bez korišćenja gotovih optimizacionih rutina.
Eksperimenti su pokazali značajan uticaj izbora stope učenja i veličine
mini-šarže na brzinu konvergencije i konačnu tačnost modela.

---

## 10. Pokretanje koda

Za pokretanje eksperimenta koristiti:

```bash
python src/main.py
