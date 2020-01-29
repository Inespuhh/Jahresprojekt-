# Jahresprojekt-
## Was zum ausführen des Codes in Python benötigt wird:
Zum starten der Skripte werden Kitematic (Alpha), Docker Quickstart Terminal, Oracle VM VirtualBox und eine Entwicklungsumgebung wie zum Beispiel Visual Studio Code benötigt.
## Anleitung zur Benutzung von Python mit Docker:
Zum öffnen der Skripte wird Visual Studio Code geöffnet
Im Terminal (CMD) muss der Pfad angegben werden, wo die Skripte liegen. Zum Beispiel bei mir C:\Users\Inija\Documents\Alle Ordner\I_Reutlingen\Jahresprojekt\Iris>
Hinter diesen Pfad wird nun 'docker ps' eingegeben und Enter gedrückt.
(docker ps kann man sich die laufenden Container anzeigen lassen.)
Wenn nichts angezeigt wird 'docker-machine stop default' eingeben und Enter drücken.
(Die VM wird gestoppt.)
Danach 'docker-machine start default' angeben und Enter drücken. 
(Die VM wird gestartet.)
(# Docker-Container können ähnlich wie virtuelle Maschinen mit den Kommandos docker pause pausiert und docker unpause wieder gestartet werden. Dies ist insbesondere sinnvoll, wenn man den Zustand eines Containers mithilfe von docker commit sichern möchte.)
Jetzt erneut 'docker ps' eingeben und Enter drücken.
(Es wird angezeigt, dass die Maschine läuft.)
'docker build --rm -f Dockerfile -t iris .' eingeben.
(docker build. Aus diesen Dockerfiles werden nun auf der Kommandozeile die beiden Docker Images erzeugt. Die Dockerfile ist auf Github geladen. In Ihr befinden sich die Module, welche mit dem Befehl 'pip install' installiert werden. 
'docker run --rm -it -p 0.0.0.0:6006:6006 iris' eingeben.
(Das Starten eines Images instanziiert einen neuen Container und wird über das Kommando docker run ausgeführt.)
Nun ist man in TensorFlow dirn. Jetzt auf das Symbol 'Remote Explorer' in der Leiste klicken.
Den gewünschten Container auswählen und auf 'Connect to Container' klicken.
Es öffnet sich ein neues Fenster, da man jetzt im Container sich befindet.
## Anleitung zur Programmierung in Python mit Docker: 
In dem Container befinden sich die Dateien wie Python Skripte, Excel-Files und CSV-Files die sich im Ordner Iris befinden.
Nun kann der Code bearbeitet werden. 
Zum ausführen des Codes muss das Terminal geöffnet werden mit der Tastenkomination 'Strg + j'.
Jetzt python und den Skriptnamen dahinter eingeben und Enter drücken.
Falls in dem Code der Befehl 'print()' existiert wird das definierte im print Befehl im Terminal ausgegeben. 
Falls es Fehlermeldungen gibt, werden diese auch im Terminal angezeigt.
Wenn das bearbeitet Skript gesichert werden soll wird es heruntergeladen und lokal auf dem Laptop gespeichert.
## Anleitung zur Visualisierung von Grafiken mit Docker:
Mit dem Befehl 'ls' wird angezeigt welche File in dem Developer liegt. 
Mit dem Befehl 'cd ./logs' wird das Verzeichnis logs angezeigt. 
Im Verzeichnis logs kann man mit dem Befehl 'ls' das erstellte Event Output angezeigt bekommen. Dies sieht zum Beispiel so aus: events.out.tfevents.1574945283.10dfde9e70aa
Dieses Event zeigt an, dass eine Grafik erstellt worden ist. 
Da mit Docker gearbeitet wird muss diese Grafik über Tensorboard visualisiert werden. Hierfür benötigt man das Event.
Mit 'cd .. ' geht man wieder aus dem logs Verzeichnis raus. Ist wieder im developer Verzeichnis. 
Das Tensorboard wird mit dem Befehl 'tensorboard --logdir=./logs' geladen. Der Vorgeschlagene Link beim ausführen funktioniert leider nicht. Man muss diesen Link 'http://192.168.99.100:6006/#images' in seinem Browser öffen. 
Im Browser sieht man nun die erstellte Grafik. 
## Nur Code lesen:
Falls man sich den Code nur ansehen will kann dies mit der hilfe von Visual Studio Code getan werden. 
