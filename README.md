# Machine Learning — Decision Trees

Assignment 1 of the Machine Learning course, part of the MSc in Internet of Things: Artificial Intelligence and Next Generation Networks at the University of the Aegean.

The assignment uses the Raisin Dataset, which contains 900 raisin grains of two varieties (Kecimen and Besni, 450 of each), described by 7 numeric morphological features. Decision trees are built with J48 (the C4.5 implementation in Weka) and CART (scikit-learn).

## Questions

1. Describe the Raisin Dataset and the problem it addresses.
2. Evaluate J48 without pruning, with rule-based pruning and with reduced-error pruning, using a 70%/30% split and 10-fold cross-validation.
3. Evaluate CART with the gini and entropy splitting criteria, without pruning and with pruning (a=0.1), using a 70%/30% split and 10-fold cross-validation.
4. Estimate the best value of the CART pruning hyperparameter a for gini, using a validation set and using 10-fold cross-validation, then compare the two approaches with each other and with Question 3.
5. Compare the results with those of Cinar et al. (2020) and identify the most competitive algorithm and settings.

This repository contains the Weka outputs for Question 2, and the code and outputs for Questions 3 and 4.

## Setup

The project uses Python 3.14. On Linux, run these commands from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

Run Question 3:

```bash
python src/cart_question3.py
```

Run Question 4:

```bash
python src/cart_question4.py
```

Run the tests:

```bash
python -m pytest tests -q
```

The two scripts overwrite their files in `results/cart/`. The Question 4 execution times change on each run, while its other results stay the same.

## Directories

- `data/`: the Raisin Dataset (`.xlsx` for Python, `.arff` for Weka, `.txt` for its description).
- `src/`: the Python scripts for Questions 3 and 4.
- `tests/`: the pytest tests for Questions 3 and 4.
- `docs/`: the assignment description (PDF, in Greek).
- `results/`: the Weka outputs (`j48_*.txt`), the Question 3 outputs (`cart/`) and the Question 4 outputs (`cart/question4/`).

---

# Machine Learning — Decision Trees

Εργασία 1 του μαθήματος Μηχανική Μάθηση, στο πλαίσιο του ΠΜΣ «Διαδίκτυο των Πραγμάτων: Τεχνητή Νοημοσύνη και Σύγχρονα Δίκτυα» του Πανεπιστημίου Αιγαίου.

Η εργασία χρησιμοποιεί το Raisin Dataset, το οποίο περιλαμβάνει 900 κόκκους σταφίδας δύο ποικιλιών (Kecimen και Besni, από 450 η καθεμία) που περιγράφονται από 7 αριθμητικά μορφολογικά χαρακτηριστικά. Τα δέντρα απόφασης δημιουργούνται με τον J48 (την υλοποίηση του C4.5 στο Weka) και τον CART (scikit-learn).

## Ερωτήματα

1. Περιγραφή του Raisin Dataset και του προβλήματος στο οποίο αναφέρεται.
2. Αξιολόγηση του J48 χωρίς pruning, με rule-based pruning και με reduced-error pruning, με διαχωρισμό 70%/30% και με 10-fold cross-validation.
3. Αξιολόγηση του CART με τα splitting criteria gini και entropy, χωρίς pruning και με pruning (a=0.1), με διαχωρισμό 70%/30% και με 10-fold cross-validation.
4. Εκτίμηση της βέλτιστης τιμής της υπερ-παραμέτρου pruning a του CART για το gini, με χρήση validation set και με χρήση 10-fold cross-validation, και σύγκριση των δύο προσεγγίσεων μεταξύ τους και με το Ερώτημα 3.
5. Σύγκριση των αποτελεσμάτων με αυτά των Cinar et al. (2020) και εντοπισμός του πιο ανταγωνιστικού αλγορίθμου και των ρυθμίσεών του.

Το αποθετήριο περιέχει τα αποτελέσματα του Weka για το Ερώτημα 2, καθώς και τον κώδικα και τα αποτελέσματα για τα Ερωτήματα 3 και 4.

## Εγκατάσταση

Χρησιμοποιείται Python 3.14. Σε Linux, εκτελέστε τις παρακάτω εντολές από τον κεντρικό φάκελο:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Εκτέλεση

Εκτέλεση του Ερωτήματος 3:

```bash
python src/cart_question3.py
```

Εκτέλεση του Ερωτήματος 4:

```bash
python src/cart_question4.py
```

Εκτέλεση των tests:

```bash
python -m pytest tests -q
```

Τα δύο scripts αντικαθιστούν τα αρχεία τους στον φάκελο `results/cart/`. Οι χρόνοι εκτέλεσης του Ερωτήματος 4 αλλάζουν σε κάθε εκτέλεση, ενώ τα υπόλοιπα αποτελέσματά του παραμένουν ίδια.

## Φάκελοι

- `data/`: το Raisin Dataset (`.xlsx` για την Python, `.arff` για το Weka, `.txt` για την περιγραφή του).
- `src/`: τα Python scripts για τα Ερωτήματα 3 και 4.
- `tests/`: τα pytest tests για τα Ερωτήματα 3 και 4.
- `docs/`: η εκφώνηση της εργασίας (PDF, στα ελληνικά).
- `results/`: τα αποτελέσματα του Weka (`j48_*.txt`), του Ερωτήματος 3 (`cart/`) και του Ερωτήματος 4 (`cart/question4/`).
