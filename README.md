# Tris-AI
Machine learning applicato al tris, senza utilizzo di librerie astratte.

-- main.py --
All'avvio del main, inizierà il processo di training del network. Sul terminale apparirà la funzione costo associata al training example appena utilizzato. Alla fine, l'utente potrà inserire un numero corrispondente ad un training example, ed ottenere l'output layer del network, e una lista contenente le mosse nell'ordine di preferenza del network. Nel file data.txt è contenuta la lista delle mosse corrette associate a ciascun training example. Al momento i training data consistono di 12 esempi, è possibile dunque testare solo gli esempi tra lo 0 e l'11. Il codice del main contiene i parametri relativi alla step size, al numero di iterazioni del training, e alla dimensione dei due layer intermedi del network.

-- Network.py --
Contiene la classe del network, con i metodi per la backpropagation e il training. Il network contiene 4 layer. Un layer di input con 18 neuroni, un layer intermedio con 4 neuroni ciascuno, un layer di output con 9 neuroni. I primi 9 neuroni di input rappresentano le mosse del network (0 per casella vuota, 1 per casella piena), gli ultimi 9 rappresentano le mosse dello sfidante. I 9 neuroni di output rappresentano la preferenza per la mossa del network (maggiore il valore, più priorità ha la mossa). Viene utilizzata una funzione di costo quadratica. Non vengono al momento applicati softmaxing (viene sempre applicata una sigmoide) e regolarizzazione. Il network non è al momento convolutivo.

-- Utilities.py --
Contiene la funzione sigmoide e la sua derivata, usate nel calcolo del network e nella backpropagation.

-- data.txt ---
Contiene i training data, e una lista leggibile delle mosse corrette associate a ciascun training data. I training data consistono in una stringa di 0 e 1. Un singolo training example consiste di uno scaglione di 27 cifre. In uno scaglione, le prime 9 rappresentano le mosse del network (0 per casella vuota, 1 per casella piena), le 9 centrali rappresentano le mosse dell'avversario, le ultime 9 rappresentano la mossa ottimale (1 sulla casella ottimale, 0 altrove).

-- TO_DO_LIST.txt --
Contiene una lista di possibili modifiche per migliorare il progetto.
