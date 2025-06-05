#import "template.typ": bsc-thesis
#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.3.2": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot, chart


#show: bsc-thesis.with(
  title: "Az epizodikus memória szerepének modellezése a rugalmas és folytonos tanulásban",
  abstract: [
    *Háttér és célkitűzés:* A biológiai rendszerek folytonos tanulási képessége alapvetően különbözik a jelenlegi mesterséges neurális hálózatok megközelítésétől. Jelen kutatás célja biológiailag inspirált, tüzelő neurális hálózatok (Spiking Neural Networks, SNN) evolúciós tervezésének elméleti és gyakorlati vizsgálata, különös tekintettel a folytonos tanulásra és az epizodikus memória szerepére.
    
    *Módszertan:* A hagyományos, előre tervezett modellek helyett evolúciós megközelítést alkalmazunk, amely neurobiológiai elveket – háromtényezős tanulási szabályok, viselkedési időskálájú szinaptikus plaszticitás (Behavioral Timescale Synaptic Plasticity, BTSP), és gátló plaszticitás – integrálva fejleszt adaptív neurális architektúrákat. Elméleti keretrendszerünk bevezeti a "genomikus szűk keresztmetszet" hipotézist, mely szerint az evolúció által preferált általános tanulási elvek csökkenthetik a keresési tér komplexitását.
    
    *Eredmények:* A JAX keretrendszerben implementált 0.1 prototípus demonstrálta az emergens viselkedés kialakulását egyszerű lokális szabályokból. Az ágens sikeresen tanult gradienskövetést 256 neuronnal, 87.65%-os súlyritkaság mellett, átlagosan 25.5 jutalmat gyűjtve epizódonként.
    
    *Következtetések:* A prototípus eredményei alátámasztják, hogy biológiailag inspirált elvek alkalmazása SNN-ekben előnyös lehet. A tervezett teljes rendszer Brian2 keretrendszerben, többfázisú környezeti progresszión keresztül optimalizálja az SNN topográfiákat, célja olyan AI rendszerek fejlesztése, amelyek többszintű adaptációval és katasztrofális felejtés nélküli folyamatos tanulással közelítik az emberi tanulás rugalmasságát.
  ],
  author: (
    name: "Ferencz Krisztián",
    program: [Molekuláris Bionika mérnöki BSc],
  ),
  supervisor: [Dr. Káli Szabolcs],
  year: [2025],
  bibliography: bibliography("refs.bib"),
)

#set math.equation(numbering: "(1)")

= A feladat ismertetése

A kutatás kezdeti célja az volt, hogy megvizsgáljuk, hogyan működnek együtt különböző memóriarendszerek – különösen az epizodikus memória – az általánosított és komplex tanulás minták, viselkedések kialakulásában. A projekt motivációja az emberi agy azon képessége volt, hogy minimális példából képes új koncepciókat elsajátítani és általánosítani @lake2015, szemben a hagyományos deep learning rendszerekkel, amelyek tipikusan nagy adatmennyiséget igényelnek.

A kutatás során azonban egy mélyebb kérdés felé fordultunk: hogyan alakulhattak ki evolúciósan azok a neurális struktúrák és tanulási mechanizmusok, amelyek lehetővé teszik a folytonos, élethosszig tartó tanulást? Ez a több találkozón és beszélgetésen át ívelő lassú perspektívaváltás vezetett egy olyan modell kidolgozásához, amely evolúciós algoritmusokkal fejleszt biológiailag inspirált neurális hálózatokat többfázisú, fejlődő környezetekben történő navigációra.

= Bevezetés

== A memóriarendszerek interakciója és az általánosítás

Az emberi agy különböző memóriarendszerei – epizodikus, szemantikus, procedurális és munkamemória – nem izoláltan működnek, hanem komplex interakciókon keresztül hozzák létre rugalmas tanulási képességeinket @eichenbaum1999. Az epizodikus memória gyors, one-shot tanulást tesz lehetővé a hippokampuszban, míg a szemantikus memória fokozatosan konszolidálja az általános tudást a neokortexben. Ez a kettős rendszer feloldhatja a stabilitás-plaszticitás dilemmát, lehetővé téve új információ elsajátítását a régi tudás elvesztése nélkül @mcclelland1995.

== A biológiai tanulás számítási modelljeinek kihívásai

A szakirodalom áttekintése során szembesültem azzal, hogy a biológiailag inspirált tanulási rendszerek modellezése rendkívül fragmentált területet képez. Számos különböző megközelítés létezik – a Hebbi plaszticitástól a spike-timing-dependent plasticity (Spike-Timing-Dependent Plasticity, STDP) mechanizmusokon át, meta-learning algoritmusokig –, de ahogy Richards és munkatársai @richards2019 áttekintésében látható, még közel sincs valódi konszenzus arról, hogy mely mechanizmusok a legfontosabbak.

Ez a felismerés vezetett a kutatási fókusz átgondolásához: ahelyett, hogy egy specifikus mechanizmust próbálnánk implementálni, egy evolúciós megközelítést választottunk, amely képes lehet felfedezni hatékony tanulási szabályokat és struktúrákat.

== Modell választás: az SNN tervezés kihívásai és előnyei

Míg az SNN-ek eredményeinek biológiai értelmezése gyakran komplex kihívást jelent, a fordított irány – biológiai eredmények alkalmazása SNN tervezésben – potenciális előnyökkel járhat. A klasszikus megközelítésekkel (pl. rekurrens neurális hálózatok, Recurrent Neural Networks, RNN) összehasonlítva az SNN-ek természetes hasonlóságot mutatnak a biológiai mechanizmusokkal: spike-alapú kommunikáció, aszinkron feldolgozás, lokális tanulási szabályok és energiahatékony működés. Ez lehetővé teszi, hogy közvetlenül alkalmazhassuk egyes neurobiológiai kutatások eredményeit – például a három-faktoros tanulási szabályokat, a különböző időskálákon működő plaszticitási mechanizmusokat, vagy a gátló interneuronok specifikus szerepeit – anélkül, hogy mesterséges absztrakciókat kellene bevezetni.

#figure(
  diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    node((0, 0), [Biológiai eredmények], shape: rect, corner-radius: 0.3em, fill: rgb("#e8f4f8")),
    node((4, 0), [SNN tervezés], shape: rect, corner-radius: 0.3em, fill: rgb("#d4e9f7")),
    node((8, 0), [Kompakt modellek], shape: rect, corner-radius: 0.3em, fill: rgb("#b8dff5")),
    edge((0, 0), (4, 0), "->", [Közvetlen alkalmazás], label-pos: 0.5),
    edge((4, 0), (8, 0), "->", [Few-shot tanulás], label-pos: 0.5),

    node((0, -2), [Klasszikus ML], shape: rect, corner-radius: 0.3em, fill: rgb("#ffe4e4")),
    node((4, -2), [Absztrakció], shape: rect, corner-radius: 0.3em, fill: rgb("#ffd4d4")),
    node((8, -2), [Nagy modellek], shape: rect, corner-radius: 0.3em, fill: rgb("#ffc4c4")),
    edge((0, -2), (4, -2), "->", [Fordítás szükséges], label-pos: 0.5, stroke: rgb("#ff6b6b")),
    edge((4, -2), (8, -2), "->", [Sok adat kell], label-pos: 0.5, stroke: rgb("#ff6b6b")),
  ),
  caption: [Biológiai elvek alkalmazásának előnye SNN-ekben klasszikus ML-hez képest],
)

Ez a közvetlen megfeleltethetőség jelentős előnyökhöz vezet: kompaktabb modellek, amelyek képesek online tanulásra minimális példából, biológiaihoz hasonlítható adaptációs mintázatokkal és általánosítási képességekkel. Míg egy RNN-ben nehéz implementálni például dopamin-modulált STDP-t vagy a dendritikus számítást, az SNN-ekben ezek természetes alapelemek lehetnek.

== A biológiai tanulás elméleti alapjai

=== Háromtényezős tanulási szabályok és neuromoduláció

A hagyományos Hebb-i tanulás, amely csak a pre- és poszt-szinaptikus aktivitást veszi figyelembe, alapvető korlátokba ütközik credit assignment és kontextusfüggő tanulás terén. A háromtényezős tanulási szabályok áttörést jelenthetenek: a szinaptikus plaszticitás nem csak a lokális neurális aktivitástól, hanem spatiálisan lokalizált, kolokált csoportokra ható, vagy globális neuromodulátoros jelektől is függ @fremaux2016. Ennek matematikai kerete:
$ Delta W = eta times E_T times M(t) $ <eq:three-factor>
ahol $E_T$ az eligibilitási nyom (eligibility trace), $M(t)$ pedig a modulátoros jel.

_Magyarázat: A @eq:three-factor egyenlet három tényező szorzataként írja le a szinaptikus súlyváltozást ($Delta W$). Az $eta$ tanulási ráta szabályozza a változás sebességét. Az $E_T$ eligibilitási nyom egy időben lassan lecsengő jel, amely „megjelöli" azokat a szinapszisokat, amelyek nemrég aktívak voltak. Az $M(t)$ modulátoros jel (pl. dopamin) pedig visszamenőlegesen megerősíti vagy gyengíti ezeket a megjelölt kapcsolatokat a kapott jutalom alapján._

Különböző neuromodulátorok specifikus funkciókat látnak el:
- *Dopamin*: jutalom-feldolgozás és megerősítéses tanulás
- *Acetilkolin*: plaszticitás modulációja, figyelem szabályozása, valamint újdonság és bizonytalanság kódolása
- *Noradrenalin és szerotonin*: arousal és adaptáció

Újabb kutatások bevezették a sejt-tipus-specifikus neuromoduláció koncepcióját, ahol a neuronok aktívan jelezhetik hozzájárulásukat a hálózat teljesítményéhez @urban-ciecko2016 @hu2014.

=== Gátló plaszticitás és hálózati stabilitás

A gátló szinaptikus plaszticitás alapvetően más elvek szerint működik, mint a serkentő. Az inhibitorikus (gátló) spike-timing-dependent plasticity (iSTDP) tipikusan szimmetrikus tanulási ablakokat mutat, ahol a potenciáció akkor következik be, ha a pre- és poszt-szinaptikus tüzelések 10 ms-on belül esnek, függetlenül a sorrendtől @vogels2011. Ez éles ellentétben áll az excitátoros STDP aszimmetrikus profilljával.

_Magyarázat: Az STDP a tüzelések időzítésén alapuló tanulási szabály. Gerjesztő szinapszisok esetén aszimmetrikus: ha a preszinaptikus neuron tüzel a posztszinaptikus előtt, erősödés (Long-Term Potentiation, LTP) történik; fordított sorrendben gyengülés (Long-Term Depression, LTD). Gátló szinapszisok esetén viszont szimmetrikus: mindkét sorrend erősítést okoz, ami stabilizálja a gerjesztés-gátlás egyensúlyt a hálózatban._

A gátló plaszticitás számos kritikus funkciót lát el:
- Fenntartja a pontos gerjesztő/gátló (Excitatory/Inhibitory, E/I) egyensúlyt
- Megakadályozza a patológikus szinkronizációt
- A hálózatot a számítás szempontjából optimális kritikalitási ponton tartja

=== Többszintű memóriakonszolidáció

A biológiai memóriarendszerek többszintű szerveződést mutatnak @kumaran2016learning. A hippokampusz gyors, mintázat-szeparált kódolást biztosít sparse reprezentációkkal, míg a neokortex lassú, a statisztikai szabályszerűségeket átfedő, elosztott reprezentációkkal operál @mcclelland1995. Az alvás-függő konszolidáció koordinálja ezeket a rendszereket: NREM (Non-Rapid Eye Movement) alvás során a hippokampális-neokortikális dinamika visszaállítja a friss memóriatracekat, míg REM (Rapid Eye Movement) alvás során a neokortex szabadon explorálja a meglévő memória attraktorokat @antony2024organizing @wang2013incorporating.

== Az evolúciós perspektíva és a genomikus szűk keresztmetszetek

Jelen projekt egyik kulcsfontosságú elméleti alapjának tekinthető a genomikus szűk keresztmetszet paradigma. Shuvaev et al. @shuvaev2024encoding szerint az állati genomok körülbelül 10#super[8] bitet tartalmaznak, míg az agyban ez a szám körülbelül 10#super[14]. Megállapításuk, hogy ez a szakadék természetes szűrőként működhet, amely a neurális struktúrák hatékony kódolását kényszerítheti ki az evolúció során.

Feltételezzük, hogy hasonló elvek működhetnek az evolúció során is: az egyszerű szenzomotoros rendszerektől a komplex kognitív képességekig vezető úton olyan általános tanulási és strukturális elvek alakulhattak ki, amelyek genomikusan kódolva, generációkon át öröklődnek és alkalmazkodnak.

== Folyamatos tanulás és a katasztrofális felejtés problémája

A biológiai rendszerek folyamatos tanulási képessége alapvetően különbözik a jelenlegi mesterséges hálózatok megközelítésétől. A komplementer tanulási rendszerek elmélete @mcclelland1995 szerint a hippokampusz és neokortex együttműködése teszi lehetővé az új információ gyors elsajátítását a régi tudás megőrzése mellett. Ez a kettős architektúra inspirálja a hierarchikus memória megközelítéseket, ahol egy gyors tanulási komponens azonnali adaptációt tesz lehetővé új mintákhoz, egy lassú konszolidációs komponens kivonja a statisztikai regularitásokat, és intelligens újrajátszási mechanizmusok biztosítják a szelektív memória reaktivációt. A projektünk célja ezen elvek evolúciós kialakulásának modellezése.

= Háttér és Kapcsolódó Munkák

A biológiailag inspirált SNN-ek és evolúciós megközelítések terén számos releváns kutatás létezik, amelyek megalapozzák és kontextualizálják a jelenlegi projektet.

== Integrált SNN rendszerek és hiperdimenziós számítás

A legújabb integrált rendszerek jelentős előrelépést mutatnak az SNN-ek képességeinek kiterjesztésében. A SpikeHD keretrendszer @deng2022memory kombinálja az SNN-eket a hiperdimenziós számítással (Hyperdimensional Computing), 15%-os pontosságjavulást érve el az önálló SNN-ekhez képest 60%-os paramétercsökkentéssel. A BI-SNN @tieck2021brain sikeresen jósolja meg az izomaktivitást EEG jelekből valós időben, demonstrálva az SNN-ek potenciálját agy-gép interfészekben. Az ASIMOV-FAM @teeter2024cognitive hippokampális struktúrákhöz hasonló gráf tanulási algoritmusokat használ kognitív térképek alkotására, ami releváns az epizodikus memória és navigáció modellezéséhez. Ezen munkák rávilágítanak az SNN-ek és más számítási paradigmák ötvözésében rejlő lehetőségekre.

== Weight Agnostic Neural Networks (WANN) inspiráció

További keresési tér redukciós lehetőséget jelent a Weight Agnostic Neural Networks (WANN) @gaier2019 megközelítése, amely a súlyok nélküli neurális hálózatok evolúciós optimalizációjára összpontosít. A WANN célja, hogy olyan architektúrákat találjon, amelyek képesek feladatokat megoldani anélkül, hogy explicit súlyértékeket kellene tanulniuk, gyakran egyetlen megosztott súlyértékkel operálva. Ez az SNN-ek esetében különösen releváns lehet, mivel:
- Az architektúra önmagában kódolhat jelentős számítási kapacitást, függetlenül a pontos súlyértékektől.
- A temporális dinamika és a spike időzítés fontosabb szerepet játszhat, mint a súlyok nagysága.
- Az evolúció természetes módon fedezhet fel ismétlődő, moduláris struktúrákat, amelyek robusztusak a súlyparaméterek változásaira.
Ez a megközelítés inspirációt adhat a genomikus szűk keresztmetszet elvének alkalmazásához, ahol az evolúció az architektúra alapvető elveit részesíti előnyben a finomhangolt paraméterek helyett.

= Módszertan

== Elméleti keretrendszer kidolgozása

A projekt során kidolgoztunk egy olyan elméleti keretrendszert, amely ötvözi:
- Evolúciós neurobiológia elveit
- Folytonos tanulás (continual learning) modern megközelítéseit
- Few-shot learning paradigmáját
- Biológiai plaszticitás számítási modelljeit

A keretrendszer központi eleme egy többszintű evolúciós folyamat, amelynek komponenseit a @tbl:evolutionary-levels táblázat foglalja össze.

#figure(
  table(
    columns: 2,
    align: (left, left),
    [*Szint*], [*Evolúciós folyamat*],
    [Környezeti evolúció], [Egyszerű szenzomotoros feladatoktól komplex kognitív kihívásokig],
    [Neurális evolúció], [Primitív reflexektől összetett tanulási képességekig],
    [Tanulási szabály evolúció], [Fix kapcsolatoktól adaptív plaszticitásig],
  ),
  caption: "Többszintű evolúciós folyamat komponensei",
) <tbl:evolutionary-levels>

== A tervezett teljes rendszer komponensei

=== Evolúciós környezet (Multi-phase Environment)

A környezet fokozatosan növekvő komplexitású fázisokból áll:
- *Fázis 1*: Egyszerű gradiens követés (kemotaxis)
- *Fázis 2*: Akadálykerülés és navigáció
- *Fázis 3*: Többcélú optimalizáció (táplálék vs. veszély)
- *Fázis 4*: Társas interakciók és kooperáció
- *Fázis N*: Komplex problémamegoldás

Minden fázis építkezik az előzőn, de új kihívásokat is bevezet, modellezve az evolúciós nyomás fokozatos növekedését. Az egyes fázisok közötti átmenetek során a neurális architektúra és a tanulási szabályok is adaptálódnak, lehetővé téve a folyamatos tanulást és alkalmazkodást. Emellett itt alkalmazzuk a korábban bevezetett "genomikus szűk keresztmetszet" elvet, amely lehetővé teheti a hatékony kódolást és a komplexitás csökkentését az evolúciós folyamat során.

=== Biológiailag inspirált neurális komponensek

==== Tüzelő neuron modellek hierarchiája

A projekt során többszintű neuron modelleket javasolunk implementálni, amelyek különböző komplexitási szinteken ragadják meg a biológiai neuronok viselkedését:

1. *Adaptív LIF (Leaky Integrate-and-Fire) neuronok*: Az alapvető Leaky Integrate-and-Fire modellt kiegészítettük adaptációs árammal, amely megragadja a spike-frekvencia adaptációt:
  $ tau_m dot(v) = -(v - E_L) - g_a (v - E_K) + I $ <eq:lif-membrane>
  $ tau_a dot(g_a) = -g_a + b sum_k delta(t - t_k) $ <eq:lif-adaptation>
  
  _Magyarázat: A @eq:lif-membrane egyenlet a neuron membránpotenciáljának ($v$) időbeli változását írja le. A $tau_m$ membrán időállandó határozza meg a változás sebességét. A $(v - E_L)$ tag a nyugalmi potenciál felé húzza a membránpotenciált, a $g_a (v - E_K)$ adaptációs áram csökkenti a gerjeszthetőséget, míg az $I$ a bejövő áram. A @eq:lif-adaptation egyenlet az adaptációs konduktancia ($g_a$) dinamikáját írja le: exponenciálisan csökken ($-g_a$), de minden tüzeléskor ($delta(t - t_k)$) megnő $b$ értékkel, így megvalósítva a tüzelési ráta adaptációt._

2. *Multi-kompartmentális modellek*: A dendritikus számítás megragadásához többrekeszes modelleket tervezhetünk, ahol különböző dendritikus ágak eltérő időskálákon működnek, lehetővé téve a többléptékű temporális feldolgozást.

3. *Heterogén sejttípusok*:
  - *Piramis sejtek*: Gerjesztő neuronok aszimmetrikus STDP-vel
  - *PV+ (Parvalbumin-positive) interneuronok*: Gyors tüzelésű gátló sejtek feedforward gátláshoz @hu2014
  - *SST+ (Somatostatin-positive) interneuronok*: Dendrit-célzó gátlás feedback kontrollhoz @urban-ciecko2016
  - *VIP+ (Vasoactive Intestinal Peptide-positive) interneuronok*: Dezinhibíciós mechanizmusok megvalósítása @pfeffer2013

==== Szinaptikus plaszticitás implementációja

A plaszticitási mechanizmusok hierarchikus szerveződést mutatnak, ahogy a @tbl:plasticity-timescales táblázatban látható:

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    [*Plaszticitás típus*], [*Időskála*], [*Funkció*],
    [Aszimmetrikus STDP @bi1998], [1-20ms], [Temporális szekvencia tanulás],
    [Szimmetrikus STDP @mishra2016], [±150ms], [Mintázat kiegészítés CA3-ban],
    [BTSP @bittner2017], [±4-8s], [Egy-próbás hely-mező kialakulás],
    [R-STDP @izhikevich2007], [3-10s], [Jutalom-vezérelt tanulás],
    [Szinaptikus skálázás @turrigiano2008], [6-48 óra], [Homeosztázis fenntartása],
  ),
  caption: "Többszintű plaszticitási mechanizmusok időskálái és funkciói",
) <tbl:plasticity-timescales>

- *Lokális plaszticitás*: STDP és homeosztátikus mechanizmusok
- *Neuromoduláció*: Dopamin, szerotonin és acetilkolin hatások szimulációja

=== Evolúciós algoritmus (CoDeepNEAT adaptáció)

Az egyelőre elméleti jelleggel elsődlegesnek megjelölt evolúciós megközelítés a következő elemeket optimalizálja:
- Kapcsolódási topológia (ki kivel kapcsolódik)
- Sejttípus eloszlás (gerjesztő/gátló arányok)
- Plaszticitási paraméterek (tanulási ráták, időállandók)
- Neuromodulációs érzékenység
- Alkalmazandó tanulási szabályok, és azok paraméterei

=== Keresési tér redukció genomikus szűk keresztmetszetekkel

A kutatás egyik fontos eleme a genomikus szűk keresztmetszetek automatikus azonosítása. Shuvaev et al. @shuvaev2024encoding genomikus hálózat (g-hálózat) architektúrája 3,1-szeres memóriacsökkentést ért el CNN (Convolutional Neural Network) súlyoknál 1%-nál kisebb pontosságvesztéssel. Feltételezzük, hogy ez a megközelítés adaptálható SNN-ek evolúciós optimalizálására.

==== Automatikus mintázat felismerés

1. *Mintázat extrakció*: Sikeres evolúciós futások elemzése információelméleti metrikákkal
  - Mutual information analízis a genotípus-fenotípus kapcsolatokra
  - Hierarchikus klaszterezés a sikeres architektúrák csoportosítására
  - Főkomponens analízis a domináns variációs tengelyek azonosítására

2. *Közös motívumok azonosítása*:
  - Generációkon át megőrződő struktúrák statisztikai szignifikanciája
  - Konvergáló evolúciós utak detektálása
  - Modularitás mérése közösség-detekciós algoritmusokkal

3. *Dimenziócsökkentés*:
  - A keresési tér redukciója a talált elvek alapján
  - Hatékony kódolási sémák leírása és implementálása
  - Kényszer-alapú optimalizáció

4. *Transzfer tanulás*:
  - A korlátozások alkalmazása új környezeti fázisokra
  - Általánosítási képesség mérése

#figure(
  diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    // Linear evolution phases
    node((0, 0), [Fázis 1:#linebreak()Gradiens követés], shape: rect, corner-radius: 0.3em, fill: rgb("#e8f4f8"), width: 3cm),
    node((5, 0), [Fázis 2:#linebreak()Akadály kerülés], shape: rect, corner-radius: 0.3em, fill: rgb("#d4e9f7"), width: 3cm),
    node((10, 0), [Fázis 3:#linebreak()Többcélú optim.], shape: rect, corner-radius: 0.3em, fill: rgb("#b8dff5"), width: 3cm),
    node((15, 0), [Fázis N:#linebreak()Komplex probléma], shape: rect, corner-radius: 0.3em, fill: rgb("#9bd3f3"), width: 3cm),

    // Genomic bottlenecks between phases
    node((2.5, -2), [Genomikus#linebreak()szűk keresztmetszet#linebreak()BN1], shape: ellipse, fill: rgb("#ffe4b5"), width: 2.5cm),
    node((7.5, -2), [Genomikus#linebreak()szűk keresztmetszet#linebreak()BN2], shape: ellipse, fill: rgb("#ffd4a1"), width: 2.5cm),
    node((12.5, -2), [Genomikus#linebreak()szűk keresztmetszet#linebreak()BN3], shape: ellipse, fill: rgb("#ffc48d"), width: 2.5cm),

    // Evolution arrows through phases
    edge((0, 0), (5, 0), "->", [Evolúció], label-pos: 0.5),
    edge((5, 0), (10, 0), "->", [Evolúció], label-pos: 0.5),
    edge((10, 0), (15, 0), "->", [Evolúció], label-pos: 0.5),

    // Bottleneck filtering
    edge((1.5, -0.3), (2.5, -1.7), "->", stroke: 1pt),
    edge((2.5, -2.3), (3.5, -0.3), "->", stroke: 1pt),
    edge((6.5, -0.3), (7.5, -1.7), "->", stroke: 1pt),
    edge((7.5, -2.3), (8.5, -0.3), "->", stroke: 1pt),
    edge((11.5, -0.3), (12.5, -1.7), "->", stroke: 1pt),
    edge((12.5, -2.3), (13.5, -0.3), "->", stroke: 1pt),

    // Transfer learning between bottlenecks
    edge((2.5, -2), (7.5, -2), "->", [Transzfer tanulás], stroke: rgb("#ff6b6b") + 2pt, label-pos: 0.5),
    edge((7.5, -2), (12.5, -2), "->", [Transzfer tanulás], stroke: rgb("#ff6b6b") + 2pt, label-pos: 0.5),

    // Information reduction annotation
    node((7.5, -4), [Információ redukció:#linebreak()10#super[14] bit → 10#super[8] bit], shape: rect, fill: rgb("#f0f0f0"), stroke: 0.5pt),
  ),
  caption: [Genomikus szűk keresztmetszetek evolúciója többfázisú környezetben. Az evolúció során a neurális architektúrák áthaladnak genomikus szűk keresztmetszeteken, amelyek információ redukcióval segítik az általános tanulási elvek kialakulását.],
)

== 0.1 Prototípus Implementációja (JAX)

A kutatás első, feltáró fázisában egy minimális, de demonstratív implementációt készítettünk JAX keretrendszerben (`meta-learning-phase-0.1.py`), amely az emergens tulajdonságok kialakulását vizsgálja egyszerű lokális szabályokból.

=== Szimulációs platform és környezet (JAX, SimpleGridWorld)
A JAX keretrendszert választottuk a gyors prototipizálás és a hardveres gyorsítás (GPU/TPU) lehetősége miatt. A környezet egy egyszerűsített `SimpleGridWorld` (mérete `GRID_WORLD_SIZE = 10`), amely Gymnasium @towers2023 API-t követ:
```python
class SimpleGridWorld:
    """Minimális grid world csak gradiens információval"""

    def __init__(self, size=10): # GRID_WORLD_SIZE
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = jnp.array([self.size // 2, self.size // 2]) # AGENT_START_DIVISOR = 2
        self.agent_heading = 0  # INITIAL_HEADING = 0 (0=N, 1=E, 2=S, 3=W)
        self.reward_pos = self._random_reward()
        self.rewards_collected = 0
        self.path_history = [tuple(self.agent_pos.tolist())]

    def get_observation(self):
        """Csak gradiens magnitúdó (irány nélkül!)"""
        dist = jnp.linalg.norm(self.reward_pos - self.agent_pos)
        max_dist = jnp.sqrt(2) * self.size # DISTANCE_POWER = 2
        gradient = 1.0 - (dist / max_dist)  # MAX_GRADIENT = 1.0. Nagyobb ha közelebb
        # Magyarázat: gradiens ∈ [0,1], ahol 1.0 = célon, 0 = max távolság
        return gradient
```

=== Hálózati architektúra és neurondinamika (0.1)
A rendszer kulcsfontosságú jellemzői a Python script `CONFIGURATION` alapján:
- *Hálózati architektúra:*
  - `GRID_SIZE = 16` neuron rács (16×16), `N_NEURONS = 256` összesen (neuron ID-k: 0-255)
  - `N_SENSORY = 1` (gradiens), `N_MOTOR = 4` (előre, balra, jobbra, marad)
  - Távolságalapú lokális kapcsolódások (`LOCAL_RADIUS = 3.0`) kis valószínűségű (`LONG_RANGE_PROB = 0.05`) hosszútávú kapcsolatokkal.
- *Neurondinamika (LIF modell):*
  - `DT = 0.5` ms, `TAU = 10.0` ms (membrán időállandó)
  - `V_REST = -65.0` mV, `V_THRESH = -55.0` mV (tüzelési küszöb)
  - `V_RESET = -65.0` mV, `REFRAC_TIME = 2.0` ms (refraktáris periódus)
  - `TAU_SYN = 5.0` ms (szinaptikus időállandó)

A hálózati állapotot egy `NamedTuple` tartalmazza:
```python
class NetworkState(NamedTuple):
    """Minimális hálózati állapot"""
    v: jnp.ndarray        # Membránpotenciálok (N_NEURONS,)
    spike: jnp.ndarray    # Jelenlegi spike-ok (N_NEURONS,)
    refrac: jnp.ndarray   # Refrakter számlálók (N_NEURONS,)
    i_syn: jnp.ndarray    # Szinaptikus bemenet (N_NEURONS,)
    w: jnp.ndarray        # Súlymátrix (N_NEURONS, N_NEURONS)
    w_in: jnp.ndarray     # Bemeneti súlyok (N_SENSORY, N_NEURONS)
    w_out: jnp.ndarray    # Kimeneti súlyok (N_NEURONS, N_MOTOR)
    trace_fast: jnp.ndarray # Gyors aktivitási nyom (N_NEURONS, TRACE_TAU = 20.0 ms)
    trace_slow: jnp.ndarray # Lassú aktivitási nyom (N_NEURONS, SLOW_TRACE_TAU = 200.0 ms)
    activity: jnp.ndarray # Közelmúltbeli aktivitási szintek (ACTIVITY_DECAY = 0.99)
```

=== Tanulási mechanizmusok (0.1)
A 0.1 modell a `local_learning_step` függvényben definiált, egyszerűsített lokális tanulási szabályokat alkalmazza (`LEARNING_RATE = 0.01`):
- Lokális koincidencia detekció (`COINCIDENCE_WEIGHT = 0.5`)
- Temporális korreláció gyors (`TRACE_TAU = 20.0` ms) és lassú (`SLOW_TRACE_TAU = 200.0` ms) tracekkel (`TEMPORAL_WEIGHT = 1.0`, `SLOW_WEIGHT = 0.2`)
- Aktivitás-függő moduláció (`MODULATION_LEARNING_SCALE = 0.1`)
- Jutalom-modulált plaszticitás a lassú tracek alapján (`REWARD_LEARNING_SCALE = 10`)
- Szinaptikus normalizáció (`MAX_INPUT_SUM = 5.0`) és motoros kimeneti súlyok tanulása (`MOTOR_LEARNING_SCALE = 5`).

A tanulási lépés implementációjának váza (a Python script alapján):
```python
@partial(jit)
def local_learning_step(state: NetworkState, reward: float):
    # Konstansok a scriptből: LEARNING_RATE, REWARD_LEARNING_SCALE, stb.
    # 1. Koincidencia detekció (COINCIDENCE_WEIGHT)
    coincidence = jnp.outer(state.spike.astype(float), state.spike.astype(float))
    # 2. Temporális korreláció (TEMPORAL_WEIGHT)
    temporal_corr = jnp.outer(state.spike.astype(float), state.trace_fast) + \
                    jnp.outer(state.trace_fast, state.spike.astype(float))
    # 3. Lassú asszociációk (SLOW_WEIGHT)
    slow_corr = jnp.outer(state.spike.astype(float), state.trace_slow)
    # 4. Aktivitás-függő moduláció (MODULATION_LEARNING_SCALE)
    modulation_factor = 1.0 + state.activity # LEARNING_MODULATION_BASE
    modulated_learning = temporal_corr * modulation_factor[:, None]
    # 5. Jutalom moduláció (REWARD_LEARNING_SCALE)
    reward_modulation = state.trace_slow[:, None] * state.trace_slow[None, :]
    dw_reward = LEARNING_RATE * reward * reward_modulation * REWARD_LEARNING_SCALE
    # Tanulási jelek kombinálása és súlyok frissítése normalizációval

    # (..)

    return state._replace(w=w_new, w_out=w_out_new)
```
Az evolúciós optimalizáció ezen fázisban még nem CoDeepNEAT alapú, hanem a lokális szabályok emergens működését vizsgáljuk rögzített, de biológiailag inspirált paraméterekkel.

=== Elvárt célok és sikerességi kritériumok (0.1)
A Python script alapján a cél `EARLY_EXIT_REWARDS = 50` jutalom gyűjtése `MAX_EPISODE_STEPS = 20000` lépés alatt. A fő cél az emergens tulajdonságok megfigyelése és a hálózati architektúra hatékonyságának értékelése ebben a minimális környezetben.

== Implementációs megfontolások a teljes rendszerhez (Brian2)

=== Szimulációs platform választása (Brian2, Brian2CUDA, MLX)
A teljes, evolúciós rendszer implementálásához a *Brian2* @stimberg2019 keretrendszert tervezzük elsődleges platformként, a következő előnyök miatt:
- Természetes differenciálegyenlet specifikáció
- Flexibilis neuron modellek
- Biológiailag realisztikus szinaptikus mechanizmusok egyszerű implementációja
- Moduláris kód generálás (C++, CUDA, stb.)

Számítási teljesítmény optimalizációra több párhuzamos megközelítést vizsgálunk:
- *Brian2CUDA*: GPU gyorsítás nagy hálózatokhoz, 10-100x teljesítménynövekedés lehetősége
- *Brian2MLX fejlesztés*: Apple platformra CUDA alapján tervezünk backend modult az MLX keretrendszerrel, a számomra elérhető eszközök ezt indokolják a gyors iteráció érdekében.
- *Felhő-alapú megoldások*: Google TPU (Tensor Processing Unit) és NVIDIA GPU erőforrások kihasználása nagyskálájú evolúciós futtatásokhoz.

= Eredmények (0.1 Prototípus)

A JAX keretrendszerben implementált 0.1 prototípus (`meta-learning-phase-0.1.py`) futtatásai során (30 epizód, `RANDOM_SEED = 42`) a következő megfigyeléseket tettük.

== Teljesítmény metrikák és tanulási görbe

A 30 epizódos tanítás során rögzített teljesítmény metrikák (lásd logs/paper/ mappa és Függelék A):
- Átlagos jutalom (utolsó 10 epizód átlaga, E30): 25.5 jutalom/epizód
- Siker arány: 0% (az `EARLY_EXIT_REWARDS = 50` jutalom elérése nem történt meg egyetlen epizódban sem a `MAX_EPISODE_STEPS = 20000` korlát miatt; a `stop_reason` jellemzően `MAXSTEPS` vagy `TIMEOUT` volt).
- Átlagos lépésszám (utolsó 10 epizód átlaga, E30): 19999 lépés/epizód.
- Legjobb epizód: A 10. epizód, 38 begyűjtött jutalommal.

// Tanulási görbe a CSV adatokból (`episode_metrics.csv`)
#let episode_data = csv("logs/paper/episode_metrics.csv")
#let episodes = episode_data.slice(1).map(row => float(row.at(0)))
#let rewards = episode_data.slice(1).map(row => float(row.at(1)))

#figure(
  canvas({
    import draw: *
    plot.plot(
      size: (8, 6),
      x-label: "Epizód",
      y-label: "Összegyűjtött jutalmak",
      x-min: 0,
      x-max: 30,
      y-min: 0,
      y-max: 40,
      legend: "inner-north-east",
      {
        // Jutalom/epizód vonal
        plot.add(
          episodes.zip(rewards),
          style: (stroke: (paint: blue, thickness: 1.5pt)),
          label: "Jutalom/epizód",
        )
        // Átlag vonal (utolsó 10 epizód alapján)
        plot.add(
          ((1, 25.5), (30, 25.5)), // A log alapján az E30 átlag 25.5
          style: (stroke: (paint: red, thickness: 1pt, dash: "dashed")),
          label: "Átlag (E30): 25.5",
        )
      },
    )
  }),
  caption: [Tanulási görbe 30 epizód során a 0.1 prototípusban. A teljesítmény nagy variabilitást mutat, az átlagos jutalom 25.5 körül stabilizálódott. A legjobb epizód (10.) 38 jutalmat ért el.],
)

#figure(
  image("logs/paper/best_episode_analysis.png", width: 100%),
  caption: [A 0.1 prototípus legjobb epizódjának (10. epizód, 38 jutalom) teljesítményelemzése a `meta-learning-phase-0.1.py` script `analyze_best_episode` funkciója által generált ábra alapján. Látható az útvonal, a jutalmak közötti idő, az útvonal hatékonysága, a látogatottsági hőtérkép és a tanulási görbe kontextusa.],
)

== Emergens hálózati tulajdonságok a 0.1 modellben

A hálózat súlyainak és aktivitásának elemzése a tanítási fázis végén (a script logja és a generált `final_network_stats.csv` alapján):
- Súly ritkasága (`weight_sparsity`): 87.65%
- Maximális súly (`max_weight`): 1.0 (telítődés egyes kapcsolatoknál)
- Átlagos súly (nemnulla súlyok átlaga, `mean_weight`): 0.081
- Átlagos aktivitás (`mean_activity`): ~7.8e-12 (nagyon alacsony baseline aktivitás)

// Hálózati statisztikák vizualizációja (`final_network_stats.csv` alapján)
#let network_stats_csv_path = "logs/paper/final_network_stats.csv"
#let network_stats = csv(network_stats_csv_path, row-type: dictionary)
}

#let metrics_map = network_stats.fold((), (acc, row) => acc + ((row.metric, float(row.value)),))

#figure(
  canvas({
    import draw: *
    chart.columnchart(
      size: (10, 6),
      x-label: "Metrikák",
      y-label: "Érték",
      value-key: 1,
      label-key: 0,
      (
        ([Ritkaság (%)], float(network_stats.at(0).value)),
        ([Átlag súly], float(network_stats.at(1).value)),
        ([Max súly], float(network_stats.at(2).value)),
      ),
      bar-style: i => (
        fill: (blue, green, red).at(i),
      ),
    )
  }),
  caption: [Hálózati teljesítmény metrikák: A súlyritkaság (87.65%) és átlagos súlyérték (0.081) együttesen energiahatékony, biológiailag plauzibilis architektúrát jeleznek. A maximális súly (1.0) telítődést mutat egyes kritikus kapcsolatoknál, ami hatékony információkódolásra utal.],
)

Megfigyelt emergens tulajdonságok (a `analyze_emergent_properties` funkció és CSV export alapján):
- *Hub neuronok kialakulása*: Bizonyos neuronok (ID-k: 54, 170, 18, 59) kiemelkedő kapcsolati számmal (`total_degree`) rendelkeztek a `hub_neurons.csv` alapján.
- *Térbeli funkcionális specializáció*: A neuronok aktivitási mintázatai és súlyeloszlásai (pl. `input_weights`, `motor_regions`) utaltak arra, hogy a rácson elfoglalt helyzetük befolyásolja szerepüket.
- *Gradiens-szelektív sejtek megjelenése*: Az `analyze_emergent_properties` funkció vizsgálja az ilyen sejteket.
- *Motor kontroll régiók differenciálódása*: A kimeneti súlyok (`w_out`) mintázata alapján kezdetleges specializáció volt látható.

// Hub neuronok kapcsoltsági eloszlása: top 5 és bottom 5 összehasonlítás
#let hub_data_csv_path = "logs/paper/hub_neurons.csv"
#let hub_data = csv(hub_data_csv_path, row-type: dictionary)
#let sorted_hubs = hub_data.sorted(key: row => -float(row.total_degree))
#let top_5_hubs = sorted_hubs.slice(0, 5)
#let bottom_5_hubs = sorted_hubs.slice(-5)
#let comparison_hubs = top_5_hubs + bottom_5_hubs
#let hub_ids = comparison_hubs.map(row => str(int(float(row.neuron_id))))
#let hub_degrees = comparison_hubs.map(row => float(row.total_degree))

#figure(
  canvas({
    import draw: *
    chart.columnchart(
      size: (12, 6),
      x-label: "Neuron ID (Top 5 | Bottom 5)",
      y-label: "Összes kapcsolat (be + ki)",
      value-key: 1,
      label-key: 0,
      hub_ids.zip(hub_degrees).map(((id, deg)) => (id, deg)),
      bar-style: i => (
        fill: if i < 5 { red } else { blue },
      ),
    )
  }),
  caption: [Hub neuron eloszlás: Top 5 legnagyobb kapcsoltságú (piros) vs. bottom 5 legkisebb kapcsoltságú (kék) neuronok. A legnagyobb és legkisebb kapcsoltságú neuronok közötti különbség mutatja a hálózat hierarchikus szerveződését és hub-szerű architektúra emergens kialakulását.],
)

== Az implementáció és elméleti keretek kapcsolata (0.1 eredmények alapján)

A 0.1 implementáció, bár egyszerűsített, már demonstrálja több kulcsfontosságú elméleti koncepció megvalósíthatóságát és emergens megjelenését:

1. *Emergens modularitás*: A térbeli elrendezés és lokális kapcsolódási szabályok, valamint a tanulási szabályok hatására funkcionális specializáció (pl. hub neuronok, potenciális szenzoros/motoros régiók) jelei mutatkoztak explicit tervezés nélkül.

2. *Aktivitás-függő moduláció*: A neuronok aktivitási szintje (`state.activity`) befolyásolja a tanulást és a szinaptikus átvitelt a `neuron_step` és `local_learning_step` függvényekben, ami egy egyszerűsített formája a biológiai neuromodulációnak.

3. *Többszintű temporális dinamika*: A gyors (`TRACE_TAU = 20.0` ms) és lassú (`SLOW_TRACE_TAU = 200.0` ms) eligibilitási tracek különböző időskálájú asszociációkat tesznek lehetővé.

4. *Genomikus szűk keresztmetszet elvek*: A minimális architektúra (`N_NEURONS = 256`) és az egyszerű lokális szabályok is képesek voltak komplex, célirányos viselkedés (gradiens követés) kialakítására. A megfigyelt magas súlyritkaság (87.65%) is erre utal, alátámasztva, hogy a biológiai korlátok és egyszerű alapelvek segíthetik a hatékony tanulási stratégiák kialakulását.

Ezek az előzetes eredmények biztatóak a teljes, Brian2 alapú evolúciós rendszer fejlesztése szempontjából.

= Diszkusszió

A 0.1 prototípus eredményei alapot szolgáltatnak a teljes evolúciós rendszerrel kapcsolatos várakozások megfogalmazásához és a kutatás tágabb kontextusba helyezéséhez.

== A 0.1 prototípus eredményeinek értékelése és továbbfejlesztési irányok (Phase 0.2)

A JAX alapú 0.1 modell (`meta-learning-phase-0.1.py`) sikeresen demonstrálta az alapvető gradienskövető viselkedés kialakulását lokális tanulási szabályok révén, és betekintést nyújtott az emergens hálózati struktúrákba (pl. hub neuronok, súlyritkaság). Az ágens átlagosan 25.5 jutalmat gyűjtött epizódonként (a tanítás végén), a legjobb epizódban 38-at, de a kitűzött 50 jutalmas sikerességi kritériumot nem érte el a 20,000 lépéses korláton belül. Ez jelzi, hogy bár alapvető tanulás történt, a stratégia még nem optimális. A következő, Phase 0.2 implementációs fázisban (mely továbbra is lehet JAX/Brian2 prototípus a teljes evolúciós rendszer előtt) tervezünk orvosolni néhány hiányosságot:

1. *Valódi háromfaktoros tanulási szabály*: A `local_learning_step` jelenlegi, egyszerűsített jutalom- és aktivitásmodulációja helyett explicit eligibilitási nyomok és globálisabb neuromodulációs jelek (pl. egyértelműbb jutalomjel) bevezetése.
2. *Javított motorikus tanulás és akcióválasztás*: A jelenlegi `decode_motor_output` zajt ad a motoros aktivitáshoz. Kifinomultabb mechanizmusok (pl. softmax alapú választás, vagy a motoros súlyok tanulásának célzottabbá tétele) javíthatják a döntéshozatalt.
3. *Részletesebb statisztikai elemzés*: A script már most is sok adatot gyűjt (`export_session_data` stb.). Ennek mélyebb, információelméleti elemzése szükséges.
4. *Brian2 migráció előkészítése*: A 0.1/0.2 tanulságainak felhasználása a biológiailag realisztikusabb Brian2 szimulációs környezetben megvalósítandó teljes evolúciós rendszer tervezéséhez.

== Várható eredmények és elméleti előrejelzések a teljes evolúciós rendszertől

A teljes, Brian2 alapú, evolúciós SNN megközelítéstől a biológiai elvek közvetlen alkalmazhatósága alapján az alábbi előnyöket és tulajdonságokat várjuk:

=== Teljesítmény projekciók és kvantitatív összehasonlítások

A @tbl:expected-advantages táblázat összefoglalja a várható előnyöket és azok elméleti alapjait:

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    [*Tulajdonság*], [*Várható előny*], [*Elméleti alap*],
    [Few-shot tanulás], [Jelentős javulás], [BTSP mechanizmus @bittner2017],
    [Online adaptáció], [Gyorsabb konvergencia], [Eligibilitási tracek @izhikevich2007],
    [Energia hatékonyság],
    [Jelentős javulás neuromorf hardveren @nazeer2025optimizing],
    [Ritka spike kódolás (>90% ritkasági arány)],

    [Paraméter hatékonyság], [Kompaktabb modellek], [Architektúra-alapú megközelítés, WANN @gaier2019],
    [Folyamatos tanulás], [Minimális felejtés], [Komplementer rendszerek @mcclelland1995],
  ),
  caption: "Várható előnyök elméleti megalapozással a teljes evolúciós rendszerben",
) <tbl:expected-advantages>

A state-of-the-art (SOTA) irodalom alapján az SNN-ek teljesítmény karakterisztikái:
1. *Energiahatékonyság*: @nazeer2025optimizing alapján az SNN-eknek 93%-nál nagyobb ritkasági arányt kell elérniük az ANN-ekhez képesti energiahatékonyság eléréséhez. Célunk ezt meghaladni az evolúciós optimalizációval.
2. *Platform-specifikus eredmények*: Neuromorf hardveren 10-100x energiacsökkentés várható ritka bemeneteknél (5% alatti aktiváció). Standard GPU-kon a cél a paritás elérése vagy javítása.
3. *Időbeli kódolás előnyei*: TTFS (Time-To-First-Spike) kódolás és más időbeli sémák alkalmazása révén jelentős késleltetés- és energiacsökkenés érhető el @park2020t2fsnet.

=== Várható emergens modularitás és funkcióspecializáció

1. *Spontán modularizáció*: Az evolúció során várhatóan funkcionálisan elkülönülő modulok alakulnak ki anélkül, hogy ezt explicit módon kódoltuk volna:
  - Szenzoros feldolgozó modulok
  - Integrációs/döntéshozó régiók
  - Motor kontroll klaszterek
  - Memória/kontextus tároló egységek (pl. hippokampusz-szerű struktúrák)

2. *Sejttípus differenciálódás*: Kezdeti, akár homogénnek tekinthető populációkból az evolúció során specializált neuron típusok emergálhatnak a feladatok és a környezeti nyomás függvényében:
  - Gyors válaszidejű "reflex" neuronok
  - Lassú integrátor sejtek komplex döntésekhez
  - Oszcillátor hálózatok időzítési feladatokhoz

=== Várható konzervatív innovációs minták (pl. exaptáció)

3. *Exaptáció*: Új képességek gyakran meglévő struktúrák újrahasznosításával jönnek létre az evolúció során. Várhatóan a modellünkben is megfigyelhetők lesznek ilyen minták:
  - Navigációs struktúrák átalakulása absztraktabb problémamegoldásra a környezet komplexitásának növekedésével
  - Szenzoros integráció mechanizmusok általánosítása új modalitásokra
  - Predikciós struktúrák adaptálása tervezési feladatokhoz

#figure(
  diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    // Exaptation flow
    node((0, 0), [Eredeti funkció], shape: rect, corner-radius: 0.3em, fill: rgb("#ffe4b5")),
    node((0, -2), [Szelekciós nyomás], shape: rect, corner-radius: 0.3em),
    node((4, -1), [Módosult struktúra], shape: rect, corner-radius: 0.3em, fill: rgb("#ffd4a3")),
    node((8, 0), [Új funkció 1], shape: rect, corner-radius: 0.3em, fill: rgb("#ffc491")),
    node((8, -2), [Új funkció 2], shape: rect, corner-radius: 0.3em, fill: rgb("#ffb380")),

    edge((0, 0), (4, -1), "->", [Exaptáció]),
    edge((0, -2), (4, -1), "->"),
    edge((4, -1), (8, 0), "->"),
    edge((4, -1), (8, -2), "->"),

    // Example
    node((0, -4), [Példa: Hippokampusz], shape: rect, corner-radius: 0.3em, fill: rgb("#e8f4f8")),
    node((4, -4), [Térbeli navigáció], shape: rect, corner-radius: 0.3em),
    node((8, -4), [Epizodikus memória], shape: rect, corner-radius: 0.3em),
    edge((0, -4), (4, -4), "->"),
    edge((4, -4), (8, -4), "->", [Exaptáció]),
  ),
  caption: [Exaptációs folyamat: meglévő struktúrák új funkciókhoz való adaptálódása az evolúció során.],
)

4. *Inkrementális komplexitás*: A fejlődés várhatóan fokozatos, építkező jellegű lesz a többfázisú környezetben:
  ```
  Fázis 1: Egyszerű reflex →
  Fázis 2: Reflex + kontextus →
  Fázis 3: Kontextuális predikció →
  Fázis N: Hierarchikus tervezés
  ```

=== Várható kritikus periódusok és plaszticitás dinamika

5. *Fejlődési ablakok*: Megfigyelhetővé válhatnak kritikus periódusok az evolúciós folyamat során, ahol a hálózat különösen fogékony a strukturális és plaszticitási változásokra, hasonlóan a biológiai fejlődéshez.
  - Korai fázis: Alapvető kapcsolódási minták és tanulási szabályok kialakulása
  - Középső fázis: Finomhangolás és specializáció az adott környezeti kihívásokhoz
  - Késői fázis: Stabilizáció és hatékonyság optimalizáció

6. *Plaszticitás-stabilitás egyensúly*: Az evolúció várhatóan automatikusan megtalálja az optimális egyensúlyt a plaszticitás (új tanulás képessége) és a stabilitás (meglévő tudás megőrzése) között.
  - Magasabb plaszticitás új vagy változó környezetekben
  - Fokozatos stabilizáció sikeres adaptáció után
  - Szelektív plaszticitás fenntartása kritikus kapcsolatoknál vagy modulokban

=== Várható skálázhatóság és általánosítás

7. *Transzfer tanulás*: Az egyszerűbb környezeti fázisokban kialakult tanulási elvek, struktúramotívumok és genomikus szűk keresztmetszetek várhatóan sikeresen alkalmazhatók és továbbfejleszthetők lesznek az összetettebb környezetekben.
  - Alapvető navigációs stratégiák → Komplex útvonaltervezés
  - Egyszerű minta felismerés → Hierarchikus kategorizáció
  - Lokális optimalizáció → Globális stratégiai tervezés

8. *Robusztusság*: A biológiai korlátok (pl. Dale-elv, energiahatékonysági kényszerek) és a genomikus szűk keresztmetszet elve paradox módon növelhetik a kifejlődött rendszerek robusztusságát:
  - Zajjal szembeni ellenállás
  - Részleges sérülések (pl. neuron kiesés) toleranciája
  - Változó környezeti feltételekhez való jobb alkalmazkodás

=== Biológiai megfeleltetések

Az evolúciós optimalizáció várhatóan a következő biológiailag plauzibilis tulajdonságokat eredményezi a kifejlődött SNN-ekben:
1. *Kapcsolatsűrűség*: A kérgi hálózatokhoz hasonló ritkasági szint kialakulása @vogels2011.
2. *Gerjesztő/gátló arány*: Dale-elv @dale1935 betartása mellett emergáló, funkcionálisan optimális E/I egyensúly.
3. *Spike aktivitás*: Energiahatékony, ritka kódolás (sparse firing) kialakulása, ami jellemző a biológiai rendszerekre.
4. *Moduláris szerveződés*: Funkcionális specializáció és moduláris architektúrák természetes emeregenciája, hasonlóan az agykéreg oszlopos vagy területi szerveződéséhez.

=== Teljesítmény projekciók és kvantitatív összehasonlítások
Bár a 38 jutalom/epizód eredmény elmaradt a 50-es céltól, a 87.65%-os súly weight_sparsity és a spontán kialakuló hub neuronok arra utalnak, hogy a hálózat hatékony belső reprezentációkat alakított ki.

=== Kvantitaív teljesítmény összehasonlítások

A sota-learning irodalom alapján az SNN-ek teljesítmény karakterisztikái:

1. *Energiahatékonyság*: @nazeer2025optimizing alapján az SNN-eknek 93%-nál nagyobb ritkasági arányt kell elérniük az ANN-ekhez képesti energiahatékonyság eléréséhez
2. *Platform-specifikus eredmények*:
  - Neuromorf hardver: 10-100x energiacsökkentés ritka bemeneteknél (5% alatti aktiváció)
  - Standard GPU-k: csak 90%-nál nagyobb ritkasággal érik el az energia paritást
3. *Időbeli kódolás előnyei*:
  - TTFS (Time-To-First-Spike) kódolás: 1,76-2,76x késleltetés javulás 27-42x energiacsökkentéssel
  - Koszinusz hasonlóság regularizációja: 0,01-0,52-vel csökkenti az optimális időlépéseket

=== Biológiai megfeleltetések

Az evolúciós optimalizáció várhatóan a következő biológiailag plauzibilis tulajdonságokat eredményezi:

1. *Kapcsolatsűrűség*: A kérgi hálózatokhoz hasonló ritkasági szint @vogels2011
2. *Gerjesztő/gátló arány*: Dale-elv @dale1935 betartása mellett emergáló egyensúly
3. *Spike aktivitás*: Energiahatékony, ritka kódolás kialakulása
4. *Moduláris szerveződés*: Funkcionális specializáció természetes emeregenciája

= Következtetések

== Főbb eredmények

A kutatás során egy ambiciózus paradigmaváltáson ment keresztül a projekt: a kezdeti memóriarendszer-interakciók vizsgálatától eljutottunk egy olyan evolúciós keretrendszer kidolgozásáig, amely képes modellezni a biológiai tanulás kialakulását és fejlődését.

1. *Elméleti keretrendszer*: egy olyan megközelítést, amely összeköti az evolúciós biológiát, a neurális fejlődést és a gépi tanulást.

2. *Genomikus bottleneck hipotézis*: Javaslatot tettem arra, hogy az evolúció során kialakult tanulási mechanizmusok általános elvei genomikus szűk keresztmetszetként szolgálhatnak, jelentősen csökkentve a keresési tér komplexitását.

3. *Többfázisú környezeti modell*: Megterveztem egy olyan szimulációs környezetet, amely az evolúciós komplexitás fokozatos növekedését modellezi.

4. *Biológiai korlátok integrációja*: A tervezett rendszer következetesen alkalmazza a neurobiológiából ismert korlátokat.

== A kutatás jelentősége

*Biológiai relevanciája*: A modell segíthet megérteni, hogyan alakultak ki az agy tanulási mechanizmusai az evolúció során, és mely elvek a legfontosabbak a rugalmas kogníció szempontjából.

*Technológiai alkalmazások*: A genomikus bottleneck-ek azonosítása új utakat nyithat hatékonyabb, biológiailag inspirált AI rendszerek fejlesztésében.

*Interdiszciplináris*: A projekt összeköti az evolúciós biológiát, a neurotudományt és a mesterséges intelligencia kutatást.

== Jövőbeli irányok

1. A teljes szimulációs környezet implementálása és validálása
2. Nagyskálájú evolúciós futtatások különböző környezeti progressziókkal
3. A talált genomikus bottleneck-ek részletes elemzése és karakterizálása
4. A modell prediktív képességeinek tesztelése valós biológiai rendszereken
5. Az elvek alkalmazása gyakorlati few-shot learning problémákra

A projekt hosszú távú víziója egy olyan általános tanulási architektúra kifejlesztése, amely nem csak reprodukálja, hanem meg is magyarázza a biológiai intelligencia rugalmasságát és adaptivitását.

== Folyamatos tanulás és a katasztrofális felejtés problémája

A biológiai rendszerek folyamatos tanulási képessége alapvetően különbözik a jelenlegi mesterséges hálózatok megközelítésétől. A komplementer tanulási rendszerek elmélete @mcclelland1995 szerint a hippokampusz és neokortex együttműködése teszi lehetővé az új információ gyors elsajátítását a régi tudás megőrzése mellett.

Ez a kettős architektúra inspirálja a hierarchikus memória megközelítéseket, ahol:
- *Gyors tanulási komponens*: Azonnali adaptáció új mintákhoz
- *Lassú konszolidációs komponens*: Statisztikai regularitások kivonása
- *Intelligens újrajátszási mechanizmus*: Szelektív memória reaktiváció

== A megközelítés korlátai és kihívásai
1. *Számítási komplexitás*:
  - Az evolúciós optimalizáció nagy populációméret és generációszám esetén rendkívül számításigényes lehet.
  - Nagyobb hálózatok (>10#super[6] neuron) szimulációja és evolúciója jelentős hardveres erőforrásokat igényel. A 0.1 prototípus (`N_NEURONS = 256`) egy 20,000 lépéses epizódja GPU-n (a script logja alapján) kb. $1453 / 30 approx 48$ másodpercet vett igénybe átlagosan, ami a skálázódásnál figyelembe veendő.
  - GPU gyorsítás nélkül egy 20,000 lépéses epizód ~5-10 percet vehet igénybe

2. *Biológiai realizmus vs. hatékonyság trade-off*:
  - Teljes biológiai hűség (pl. dendritikus számítás, ioncsatorna dinamika) számításilag kivitelezhetetlen
  - Leegyszerűsített modellek használata csökkentheti a prediktív erőt
  - Az absztrakciós szint megválasztása kritikus a sikerhez

3. *Validálási és összehasonlíthatósági kihívások*:
  - In vivo neurális adatokkal való közvetlen összehasonlítás nehéz
  - A biológiai variabilitás modellezése további komplexitást jelent
  - Standardizált benchmark-ok hiánya SNN few-shot tanuláshoz

4. *Skálázási és általánosítási problémák*:
  - A kis hálózatokban (256 neuron) talált elvek nem feltétlenül skálázódnak lineárisan
  - A genomikus bottleneck-ek környezetfüggőek lehetnek
  - Cross-domain transzfer tanulás validálása szükséges

5. *Elméleti korlátok*:
  - A háromfaktoros tanulási szabályok optimális paraméterei ismeretlenek
  - Az emergens tulajdonságok prediktálhatósága limitált
  - A credit assignment probléma csak részben megoldott

6. *Implementációs kihívások*:
  - Brian2 és JAX közötti interoperabilitás
  - Reprodukálhatóság biztosítása sztochasztikus rendszerekben
  - Hibakeresés emergens rendszerekben komplex

== Jövőbeli irányok

A kutatás folytatása három fő irányban tervezett:

*Rövid távon (Phase 0.2 és Brian2 előkészítés)*:
- A 0.1 JAX implementáció továbbfejlesztése (Phase 0.2): Valódi háromfaktoros tanulási szabályok bevezetése, javított motorikus kontroll és akcióválasztás, valamint a 0.1 során gyűjtött adatok mélyebb elemzése.
- A Brian2 keretrendszerre való áttérés előkészítése: A 0.1/0.2 tanulságainak felhasználásával a neuron- és szinapszis modellek, valamint a tanulási szabályok implementálása Brian2-ben. Egyszerűbb evolúciós algoritmusok tesztelése kisebb Brian2 modelleken.

*Közép távon*: Genomikus szűk keresztmetszetek automatikus azonosítása és karakterizációja az evolúciós futtatásokból. A talált elvek alkalmazása komplex navigációs és tanulási feladatokra. Skálázás nagyobb hálózatokra (10#super[4]-10#super[6] neuron) GPU gyorsítással.

*Hosszú távon*: Általános tanulási architektúra kifejlesztése, amely ötvözi a few-shot tanulást, folyamatos adaptációt és energiahatékonyságot. A biológiai és mesterséges intelligencia közötti híd megteremtése, amely új utakat nyithat mind az idegtudományi megértés, mind a gyakorlati alkalmazások terén.

A projekt célja demonstrálni, hogy a biológiai elvek közvetlen alkalmazása SNN-ekben nemcsak lehetséges, hanem előnyös is lehet a hagyományos megközelítésekkel szemben.

= Reprodukálhatóság és Kód Elérhetőség

== Szimulációs környezet
A teljes implementáció nyílt forráskódú és elérhető:
- GitHub repository: https://github.com/krisztiaan/evo-snn
- Fő kód: `meta-learning-phase-0.1.py` (JAX keretrendszer)
- Futási környezet: Python 3.8+, JAX, NumPy, Matplotlib
- Hardware követelmények: GPU ajánlott (20,000 lépés/epizód ~48s GPU-n vs. ~5-10 perc CPU-n)

== Adatok és Log Fájlok
Az eredmények reprodukálásához szükséges adatok a repository-ban:
- Training logs: `logs/paper/` mappa
- CSV exports: `episode_metrics.csv`, `hub_neurons.csv`, `final_network_stats.csv`
- Vizualizációk: `best_episode_analysis.png`, `learning_curve.csv`

== Konfigurációs Paraméterek
Kulcs paraméterek a reprodukáláshoz:
- `RANDOM_SEED = 42` (determinisztikus eredmények)
- `N_NEURONS = 256`, `GRID_SIZE = 16`
- `MAX_EPISODE_STEPS = 20000`, `EARLY_EXIT_REWARDS = 50`
- `LEARNING_RATE = 0.01`, `TAU = 10.0` ms

= Köszönetnyilvánítás

Köszönöm Dr. Káli Szabolcsnak a folyamatos támogatást és értékes útmutatást a kutatás során, különösen az elmúlt év során nyújtott türelmét és rendkívül értékes beszélgetéseket.
}
