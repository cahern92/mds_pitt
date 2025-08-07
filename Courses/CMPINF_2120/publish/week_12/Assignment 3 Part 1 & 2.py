{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c4afbe22-2067-4546-84e7-70fc3e42a998",
   "metadata": {},
   "source": [
    "## Assignment 3 Part 1\n",
    "### Import Modules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f4dec6bb-2934-4854-8910-bf817790fff7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot\n",
    "\n",
    "from sklearn.model_selection import train_test_split, KFold\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.utils import resample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0e4dd2d5-16ef-455e-aeda-47dd91ea6683",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Linear Regression X and Y\n",
    "df = pd.read_csv('APMM9-dataset24.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5d81a32f-59b9-4c3f-bfd9-ab1ba89fb954",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 2000 entries, 0 to 1999\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   X       2000 non-null   float64\n",
      " 1   y       2000 non-null   float64\n",
      "dtypes: float64(2)\n",
      "memory usage: 31.4 KB\n"
     ]
    }
   ],
   "source": [
    "df.head()\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b26bab05-53e8-4479-941c-0a6f57b82cee",
   "metadata": {},
   "outputs": [],
   "source": [
    "traintest_list = [.60, .70, .80, .90, .95]\n",
    "# Define test sizes\n",
    "test_sizes = [0.4, 0.3, 0.2, 0.1, 0.05]\n",
    "rmse_list = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a1779576-d4c1-40e0-a37e-32836c38888a",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[[\"X\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f30e749c-1f7e-4582-9dc7-ce86e3db8ef9",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df[[\"y\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3dfa3776-12fd-4047-a5ed-5ac8d62461a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q1: Average RMSE over all splits: 22.848\n",
      "Q2: Standard Deviation of RMSE over all splits: 0.188\n"
     ]
    }
   ],
   "source": [
    "# for num in test_sizes:\n",
    "#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num, random_state = 2120)\n",
    "# # for num in traintest_list:\n",
    "# #     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num, random_state = 2120)\n",
    "\n",
    "#     scaler = StandardScaler()\n",
    "#     X_train = scaler.fit_transform(X_train)   # Fit scaler only on train\n",
    "#     X_test = scaler.transform(X_test)         # Use same scaler to transform test\n",
    "\n",
    "#     model = LinearRegression().fit(X_train, y_train)\n",
    "#     predict = model.predict(X_test)\n",
    "#     RMSE = np.sqrt(np.mean((predict - y_test)**2))\n",
    "#     np.sqrt(mean_squared_error(predict, y_test))\n",
    "    \n",
    "#     cesar = {\n",
    "#         \"train_test_size\": round(num, 2),\n",
    "#         \"test_size\": round(1 - num, 2),\n",
    "#         \"RMSE\": RMSE\n",
    "#     }\n",
    "#     results.append(cesar)\n",
    "for test_size in test_sizes:\n",
    "    # Split\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X, y, test_size=test_size, random_state=2120\n",
    "    )\n",
    "    \n",
    "    # Scale\n",
    "    scaler = StandardScaler()\n",
    "    X_train_scaled = scaler.fit_transform(X_train)\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "    \n",
    "    # Fit model\n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train_scaled, y_train)\n",
    "    \n",
    "    # Predict and evaluate\n",
    "    y_pred = model.predict(X_test_scaled)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "    rmse_list.append(rmse)\n",
    "\n",
    "# Average RMSE\n",
    "avg_rmse = round(np.mean(rmse_list), 3)\n",
    "print(\"Q1: Average RMSE over all splits:\", avg_rmse)\n",
    "\n",
    "std_mse = round(np.std(rmse_list), 3)\n",
    "print(\"Q2: Standard Deviation of RMSE over all splits:\", std_mse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a3ecc78a-0152-4efe-9358-ec622bf0d5ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "def cvfolds(fold, q):\n",
    "    kf = KFold(n_splits=fold, shuffle=False)\n",
    "    rmse_folds = []\n",
    "    \n",
    "    for train_index, test_index in kf.split(X):\n",
    "        X_train, X_test = X.iloc[train_index], X.iloc[test_index]\n",
    "        y_train, y_test = y.iloc[train_index], y.iloc[test_index]\n",
    "    \n",
    "        # Standardize\n",
    "        scaler = StandardScaler()\n",
    "        X_train_scaled = scaler.fit_transform(X_train)\n",
    "        X_test_scaled = scaler.transform(X_test)\n",
    "    \n",
    "        # Train model\n",
    "        model = LinearRegression()\n",
    "        model.fit(X_train_scaled, y_train)\n",
    "    \n",
    "        # Predict\n",
    "        y_pred = model.predict(X_test_scaled)\n",
    "        rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "        rmse_folds.append(rmse)\n",
    "    \n",
    "    avg_rmse_folds = round(np.mean(rmse_folds), 3)\n",
    "\n",
    "    # -------------------------------\n",
    "    # Results\n",
    "    # -------------------------------\n",
    "    print(f\"{q}: Average RMSE over {fold}-Fold CV:\", avg_rmse_folds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5cddd115-08cf-4d87-b1e6-fb1bd54f2f62",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q3: Average RMSE over 5-Fold CV: 23.972\n"
     ]
    }
   ],
   "source": [
    "# Q3\n",
    "cvfolds(5, \"Q3\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "eca2e333-e697-4e76-940f-0c2828f0824a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q4: Average RMSE over 10-Fold CV: 23.943\n"
     ]
    }
   ],
   "source": [
    "# Q4 with 10-Folds\n",
    "cvfolds(10, \"Q4\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0ad2adee-d9e1-4b2d-b347-8bea42d73024",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q5: Average RMSE over 2000-Fold CV: 19.194\n"
     ]
    }
   ],
   "source": [
    "# Q5\n",
    "cvfolds(len(X), \"Q5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "37a9a0ed-9cd2-4979-b32f-88d2aa1c173b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import LeaveOneOut"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e18efcf1-ca1a-48c0-b8b9-f1a5dc2c05b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average RMSE over LOOCV: 19.194\n"
     ]
    }
   ],
   "source": [
    "# Set up LOOCV\n",
    "loo = LeaveOneOut()\n",
    "rmse_list = []\n",
    "\n",
    "for train_index, test_index in loo.split(X):\n",
    "    X_train, X_test = X.iloc[train_index], X.iloc[test_index]\n",
    "    y_train, y_test = y.iloc[train_index], y.iloc[test_index]\n",
    "\n",
    "    # Standardize\n",
    "    scaler = StandardScaler()\n",
    "    X_train_scaled = scaler.fit_transform(X_train)\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "    # Train model\n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train_scaled, y_train)\n",
    "\n",
    "    # Predict\n",
    "    y_pred = model.predict(X_test_scaled)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "    rmse_list.append(rmse)\n",
    "\n",
    "# Average RMSE over LOOCV\n",
    "avg_rmse_loocv = round(np.mean(rmse_list), 3)\n",
    "print(\"Average RMSE over LOOCV:\", avg_rmse_loocv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "93656f27-cf85-49b1-b9ca-2ea946347c45",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q6 - Bootstrapping (n=100)\n",
    "def boot(n, q):\n",
    "    # Bootstrapping\n",
    "    num_iterations = n\n",
    "    rmse_list = []\n",
    "    \n",
    "    n_samples = len(X)\n",
    "    \n",
    "    for boot in range(num_iterations):\n",
    "        # Sample with replacement\n",
    "        boot_sample = resample(df, replace=True, n_samples=n_samples, random_state=10 * boot)\n",
    "        \n",
    "        # Get OOB samples\n",
    "        boot_indices = boot_sample.index\n",
    "        oob_indices = list(set(df.index) - set(boot_indices))\n",
    "        \n",
    "        if len(oob_indices) == 0:\n",
    "            continue  # skip if no OOB data (rare, but possible)\n",
    "        \n",
    "        oob_data = df.loc[oob_indices]\n",
    "        \n",
    "        # Separate features and target\n",
    "        X_train = boot_sample.drop(columns=['y'])\n",
    "        y_train = boot_sample['y']\n",
    "        X_test = oob_data.drop(columns=['y'])\n",
    "        y_test = oob_data['y']\n",
    "        \n",
    "        # Train and predict\n",
    "        model = LinearRegression()\n",
    "        model.fit(X_train, y_train)\n",
    "        y_pred = model.predict(X_test)\n",
    "        \n",
    "        # Compute RMSE\n",
    "        rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "        rmse_list.append(rmse)\n",
    "    \n",
    "    # Final average RMSE over all bootstrap OOB test sets\n",
    "    avg_rmse_bootstrap = round(np.mean(rmse_list), 3)\n",
    "    print(f\"{q}: Average RMSE over {n} bootstraps (OOB testing):\", avg_rmse_bootstrap)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6c6ec2e9-f803-4d6f-833b-dc1a020b4c50",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q6: Average RMSE over 100 bootstraps (OOB testing): 23.967\n"
     ]
    }
   ],
   "source": [
    "# Q6\n",
    "boot(100, \"Q6\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c6b062d4-fb49-42cb-bd08-9fe905362adc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q7: Average RMSE over 200 bootstraps (OOB testing): 23.985\n"
     ]
    }
   ],
   "source": [
    "# Q7 Bootstrapping with 200 iterations\n",
    "boot(200, \"Q7\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "fe7435da-f2e3-40d8-a237-626b78a6e784",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q8: Average RMSE over 400 bootstraps (OOB testing): 23.992\n"
     ]
    }
   ],
   "source": [
    "# Q8 Bootstrapping with 400 iterations\n",
    "boot(400, \"Q8\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c10908c4-3d8a-4999-9a9f-6e91d26446ad",
   "metadata": {},
   "source": [
    "### Classification Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "759bc360-00d3-4e4d-8b4c-21c6a2934bdd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df32 = pd.read_csv('APMM9-dataset32.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "019ee187-1fab-4943-b094-076177e4dca6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 500 entries, 0 to 499\n",
      "Data columns (total 2 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   actual     500 non-null    object\n",
      " 1   predicted  500 non-null    object\n",
      "dtypes: object(2)\n",
      "memory usage: 7.9+ KB\n"
     ]
    }
   ],
   "source": [
    "df32.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "3d6f5a3a-c25b-4136-b961-8cec4626db6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>actual</th>\n",
       "      <th>predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>label1</td>\n",
       "      <td>label1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>label4</td>\n",
       "      <td>label4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>label2</td>\n",
       "      <td>label2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>label2</td>\n",
       "      <td>label1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>label1</td>\n",
       "      <td>label1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   actual predicted\n",
       "0  label1    label1\n",
       "1  label4    label4\n",
       "2  label2    label2\n",
       "3  label2    label1\n",
       "4  label1    label1"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "62836174-585c-4f3d-8282-490187bb0e44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual\n",
       "label1    241\n",
       "label2    128\n",
       "label3     76\n",
       "label4     55\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32.actual.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "07a9b84c-0d62-4e87-ac9a-8d0787d21913",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "predicted\n",
       "label1    185\n",
       "label2    137\n",
       "label4     92\n",
       "label3     86\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32.predicted.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "e6896a73-0b25-4873-9ced-b57d32551f07",
   "metadata": {},
   "outputs": [],
   "source": [
    "dict_lab = {\n",
    "    \"label1\": 1,\n",
    "    \"label2\": 2,\n",
    "    \"label3\": 3,\n",
    "    \"label4\": 4\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "ccc49cfd-a294-4f4f-8dcd-9d3d3b9733ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "labels = [\"label1\", \"label2\", \"label3\", \"label4\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "1247d58e-bd66-4880-8c68-ce05c6aa538f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df32['actnum'] = df32['actual'].map(dict_lab)\n",
    "df32['prednum'] = df32['predicted'].map(dict_lab)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "da237b9e-2455-436b-a0f2-2dc4c9654e1d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>actual</th>\n",
       "      <th>predicted</th>\n",
       "      <th>actnum</th>\n",
       "      <th>prednum</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>label1</td>\n",
       "      <td>label1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>label4</td>\n",
       "      <td>label4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>label2</td>\n",
       "      <td>label2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>label2</td>\n",
       "      <td>label1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>label1</td>\n",
       "      <td>label1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>495</th>\n",
       "      <td>label1</td>\n",
       "      <td>label1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>496</th>\n",
       "      <td>label2</td>\n",
       "      <td>label4</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>497</th>\n",
       "      <td>label1</td>\n",
       "      <td>label1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>498</th>\n",
       "      <td>label4</td>\n",
       "      <td>label2</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>499</th>\n",
       "      <td>label4</td>\n",
       "      <td>label4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>500 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     actual predicted  actnum  prednum\n",
       "0    label1    label1       1        1\n",
       "1    label4    label4       4        4\n",
       "2    label2    label2       2        2\n",
       "3    label2    label1       2        1\n",
       "4    label1    label1       1        1\n",
       "..      ...       ...     ...      ...\n",
       "495  label1    label1       1        1\n",
       "496  label2    label4       2        4\n",
       "497  label1    label1       1        1\n",
       "498  label4    label2       4        2\n",
       "499  label4    label4       4        4\n",
       "\n",
       "[500 rows x 4 columns]"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "9adc0499-c739-4eb6-b07f-e42a75164acf",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(df32.actnum, df32.prednum)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "409eea90-0a42-4d98-8d92-ede0961350a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[159,  32,  21,  29],\n",
       "       [ 14,  92,  10,  12],\n",
       "       [  7,   8,  54,   7],\n",
       "       [  5,   5,   1,  44]])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7c8a77ac-a36f-4f27-ac45-a26931910754",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q9\n",
    "tp_label1 = cm[0, 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "01a41589-a9cb-425a-9e6a-6473c71ca1fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# For Label 1\n",
    "df32[\"act_lab1\"] = np.where(df32.actual == \"label1\", 1, 0)\n",
    "df32[\"pred_lab1\"] = np.where(df32.predicted == \"label1\", 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "cfd4c129-b93f-4c25-bc9f-d9ac7a3ba71f",
   "metadata": {},
   "outputs": [],
   "source": [
    "TN, FP, FN, TP = confusion_matrix(df32.act_lab1, df32.pred_lab1).ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "89deaddf-2b39-4aeb-96f1-1200e617155a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "precision = precision_score(df32.act_lab1, df32.pred_lab1)\n",
    "recall = recall_score(df32.act_lab1, df32.pred_lab1)\n",
    "f1 = f1_score(df32.act_lab1, df32.pred_lab1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "c3a88e4c-ea61-4b47-aeaf-8a656f7d665a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TP: 159\n",
      "FP: 26\n",
      "TN: 233\n",
      "FN: 82\n",
      "(Precision: 0.8594594594594595\n",
      "(Recall: 0.6597510373443983\n",
      "(F1 Score: 0.7464788732394365\n"
     ]
    }
   ],
   "source": [
    "print(f\"TP: {TP}\")\n",
    "print(f\"FP: {FP}\")\n",
    "print(f\"TN: {TN}\")\n",
    "print(f\"FN: {FN}\")\n",
    "print(f\"(Precision: {precision}\")\n",
    "print(f\"(Recall: {recall}\")\n",
    "print(f\"(F1 Score: {f1}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b099cfde-6e70-45ae-a1b9-47886ec3f334",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True Positives for label1: 159\n"
     ]
    }
   ],
   "source": [
    "print(\"True Positives for label1:\", tp_label1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "39f937b4-12af-4802-b37c-7c646fd19a50",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       159\n",
       "predicted    159\n",
       "actnum       159\n",
       "prednum      159\n",
       "act_lab1     159\n",
       "pred_lab1    159\n",
       "dtype: int64"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[(df32.actual==\"label1\") & (df32.predicted ==\"label1\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "c7927542-4db3-4cab-a765-fc2e47ce7ce6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q10\n",
    "tp_label2 = cm[1, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "7d25eb6a-7e87-4834-b7c5-e4a7ada1a4d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True Positives for label2: 92\n"
     ]
    }
   ],
   "source": [
    "print(\"True Positives for label2:\", tp_label2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "dc9c29f3-508c-4020-8915-6d0d99e0420c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       92\n",
       "predicted    92\n",
       "actnum       92\n",
       "prednum      92\n",
       "act_lab1     92\n",
       "pred_lab1    92\n",
       "dtype: int64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[(df32.actual==\"label2\") & (df32.predicted ==\"label2\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "292288a4-7595-4df5-bd77-31fccca0441c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q11\n",
    "tp_label3 = cm[2, 2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "cc2a8d0b-3868-43c8-8cf4-9976ba8434a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True Positives for label3: 54\n"
     ]
    }
   ],
   "source": [
    "print(\"True Positives for label3:\", tp_label3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a61bd3cc-631b-41ca-8731-ff1cad094037",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       54\n",
       "predicted    54\n",
       "actnum       54\n",
       "prednum      54\n",
       "act_lab1     54\n",
       "pred_lab1    54\n",
       "dtype: int64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[(df32.actual==\"label3\") & (df32.predicted ==\"label3\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "1f7b9175-c060-4fa7-831b-3055cdfb444f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q12\n",
    "tp_label4 = cm[3, 3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "5cf879c5-a0b1-4ab3-95c0-461551b59911",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True Positives for label4: 44\n"
     ]
    }
   ],
   "source": [
    "print(\"True Positives for label4:\", tp_label4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "ea394307-e26c-4522-aa65-aca6f0a9dcba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       44\n",
       "predicted    44\n",
       "actnum       44\n",
       "prednum      44\n",
       "act_lab1     44\n",
       "pred_lab1    44\n",
       "dtype: int64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[(df32.actual==\"label4\") & (df32.predicted ==\"label4\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "ce365b72-eb7d-483d-a187-e94f9ccaf485",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q13 FP for Label1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "665d5507-4edf-4f4e-a4c0-20e12172f5de",
   "metadata": {},
   "outputs": [],
   "source": [
    "fp_label1 = cm[1, 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "4ead30c4-f674-4e2c-a249-914c23209182",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fase Positives for label1: 14\n"
     ]
    }
   ],
   "source": [
    "print(\"Fase Positives for label1:\", fp_label1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "175c17f6-4107-4006-ad40-ebf76554bd1b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       26\n",
       "predicted    26\n",
       "actnum       26\n",
       "prednum      26\n",
       "act_lab1     26\n",
       "pred_lab1    26\n",
       "dtype: int64"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[~(df32.actual==\"label1\") & (df32.predicted ==\"label1\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "bdabf830-4c41-42c3-825d-2d1c74603de0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q14 FP for Label2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "2ce41e83-597a-440b-8a0f-1a3136728066",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       45\n",
       "predicted    45\n",
       "actnum       45\n",
       "prednum      45\n",
       "act_lab1     45\n",
       "pred_lab1    45\n",
       "dtype: int64"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[~(df32.actual==\"label2\") & (df32.predicted ==\"label2\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "bdf483bc-826c-42b9-a8c7-9824fe4d336a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q15 FP for Label3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "72a29a5b-84cc-4f93-95a4-11b4f3496256",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       32\n",
       "predicted    32\n",
       "actnum       32\n",
       "prednum      32\n",
       "act_lab1     32\n",
       "pred_lab1    32\n",
       "dtype: int64"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[~(df32.actual==\"label3\") & (df32.predicted ==\"label3\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "0129912e-f12e-4b5d-8412-e2e75e8a86f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q16 FP for Label4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "d137d843-18b8-4461-b7cd-c65534d47611",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       48\n",
       "predicted    48\n",
       "actnum       48\n",
       "prednum      48\n",
       "act_lab1     48\n",
       "pred_lab1    48\n",
       "dtype: int64"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[~(df32.actual==\"label4\") & (df32.predicted ==\"label4\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "86bb59f8-1dc8-4ddf-8607-bfd62806790d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q17 FN for Label1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "a9c1ccc4-ab0c-4a9f-a83a-7102b518772d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "actual       82\n",
       "predicted    82\n",
       "actnum       82\n",
       "prednum      82\n",
       "act_lab1     82\n",
       "pred_lab1    82\n",
       "dtype: int64"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df32[(df32.actual==\"label1\") & ~(df32.predicted ==\"label1\")].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "33d039d8-8951-4935-a6cd-11cf1dea5f87",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class 1: TP=159, FP=26, FN=82, TN=233, Precision=0.85946, Recall=0.65975, F1_Score=0.74648\n",
      "Class 2: TP=92, FP=45, FN=36, TN=327, Precision=0.67153, Recall=0.71875, F1_Score=0.69434\n",
      "Class 3: TP=54, FP=32, FN=22, TN=392, Precision=0.62791, Recall=0.71053, F1_Score=0.66667\n",
      "Class 4: TP=44, FP=48, FN=11, TN=397, Precision=0.47826, Recall=0.80000, F1_Score=0.59864\n"
     ]
    }
   ],
   "source": [
    "def multiclass_metrics(cm):\n",
    "    n_classes = cm.shape[0]\n",
    "    metrics = {}\n",
    "\n",
    "    for i in range(n_classes):\n",
    "        TP = cm[i, i]\n",
    "        FP = cm[:, i].sum() - TP\n",
    "        FN = cm[i, :].sum() - TP\n",
    "        TN = cm.sum() - (TP + FP + FN)\n",
    "\n",
    "        Precision = TP / (TP + FP) if (TP + FP) != 0 else 0\n",
    "        Recall = TP / (TP + FN) if (TP + FN) != 0 else 0\n",
    "        F1_Score = (2*(Precision*Recall)/(Precision + Recall)) if (Precision + Recall) != 0 else 0\n",
    "\n",
    "        metrics[f\"Class {i+1}\"] = {\n",
    "            \"TP\": TP,\n",
    "            \"FP\": FP,\n",
    "            \"FN\": FN,\n",
    "            \"TN\": TN,\n",
    "            \"Precision\": Precision,\n",
    "            \"Recall\": Recall,\n",
    "            \"F1_Score\": F1_Score\n",
    "        }\n",
    "    return metrics\n",
    "\n",
    "# Example usage:\n",
    "metrics = multiclass_metrics(cm)\n",
    "for label, vals in metrics.items():\n",
    "    print(f\"{label}: TP={vals['TP']}, FP={vals['FP']}, FN={vals['FN']}, TN={vals['TN']}, Precision={vals['Precision']:.5f}, Recall={vals['Recall']:.5f}, F1_Score={vals['F1_Score']:.5f}\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "dca804d2-3953-4067-9515-38825d847fef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q24: Macro averaged Precision\n",
    "precisions = [vals[\"Precision\"] for vals in metrics.values()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "99ed72d5-074b-4ca9-affe-86a04126a4ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "macro_precision = sum(precisions) / len(precisions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "7d89d1b3-5d2a-4d7a-a8f1-86a236854e2e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Macro-averaged Precision: 0.65929\n"
     ]
    }
   ],
   "source": [
    "print(f\"Macro-averaged Precision: {macro_precision:.5f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "5b843a75-69e8-42cb-b734-a3114600eeaf",
   "metadata": {},
   "outputs": [],
   "source": [
    "macro_precision = precision_score(df32['actual'], df32['predicted'], average='macro')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "ce51c4da-6f17-426f-8310-7f5ddbf25c34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6592900381210478"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "macro_precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "c4acebd5-c7c6-4f05-bba1-d2b7533a2e8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q25: Macro averaged Recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "6b2e221b-1245-4699-adae-f1bee7f885c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "recalls = [vals[\"Recall\"] for vals in metrics.values()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "8973dd54-1b1b-4d44-8ae8-110e6d09999b",
   "metadata": {},
   "outputs": [],
   "source": [
    "macro_recall = sum(recalls) / len(recalls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "60ebe7e3-6d02-4986-a3be-aa687be2c5d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Macro-averaged Recall: 0.72226\n"
     ]
    }
   ],
   "source": [
    "print(f\"Macro-averaged Recall: {macro_recall:.5f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "4c914afa-3154-4d48-beab-67140ba63913",
   "metadata": {},
   "outputs": [],
   "source": [
    "macro_recall = recall_score(df32['actual'], df32['predicted'], average='macro')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "e88f3453-3fe1-4c4b-90e9-6a8caed0197c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.722256838283468"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "macro_recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "c2db3ebf-b812-43ba-800d-bbad6897c0ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q26-27: Weight Averaged Precision and Recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "58361ace-10ad-4667-9524-7033321b32fc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Weighted Precision: 0.73422\n",
      "Weighted Recall: 0.69800\n"
     ]
    }
   ],
   "source": [
    "def weighted_avg_metrics(cm):\n",
    "    n_classes = cm.shape[0]\n",
    "    supports = cm.sum(axis=1)  # true samples per class (row sums)\n",
    "    metrics = multiclass_metrics(cm)  # from previous function\n",
    "\n",
    "    weighted_precision = sum(metrics[f\"Class {i+1}\"]['Precision'] * supports[i] for i in range(n_classes)) / supports.sum()\n",
    "    weighted_recall = sum(metrics[f\"Class {i+1}\"]['Recall'] * supports[i] for i in range(n_classes)) / supports.sum()\n",
    "\n",
    "    print(f\"Weighted Precision: {weighted_precision:.5f}\")\n",
    "    print(f\"Weighted Recall: {weighted_recall:.5f}\")\n",
    "\n",
    "# Example usage:\n",
    "weighted_avg_metrics(cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "ac213fde-9467-4794-ab99-5284dd2e30a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "weight_precision = precision_score(df32['actual'], df32['predicted'], average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "33b76e3f-97fc-45a6-8683-f6cd49336497",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7342224243358737"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "weight_precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "2f5cab63-c136-412f-b115-75f8ffd0009e",
   "metadata": {},
   "outputs": [],
   "source": [
    "weight_recall = recall_score(df32['actual'], df32['predicted'], average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "01755823-d563-4594-8c11-4211bf0967a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.698"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "weight_recall"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b72815c3-eac4-4eba-908e-c0d122b71431",
   "metadata": {},
   "source": [
    "## Assignment 3 Part 2\n",
    "### Import Modules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "8f724fa0-80ba-44fd-a7b1-8fee0f41ba88",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import VotingClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ec56dc5d-cd3e-4ec8-9c3d-fde2a9a3cc20",
   "metadata": {},
   "source": [
    "### Loading Dataset 11"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "dff8f792-6d4a-40dc-a347-914482cf380a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df11 = pd.read_csv(\"APMM10-clf-dataset11.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "b12af7c2-684c-47a4-be0c-6ef682b1e4b9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 500 entries, 0 to 499\n",
      "Data columns (total 3 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   X1      500 non-null    float64\n",
      " 1   X2      500 non-null    float64\n",
      " 2   y       500 non-null    int64  \n",
      "dtypes: float64(2), int64(1)\n",
      "memory usage: 11.8 KB\n"
     ]
    }
   ],
   "source": [
    "df11.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "fe2f8e45-db3e-4cfc-91ca-0e2b3a16d3f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>X1</th>\n",
       "      <th>X2</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.854774</td>\n",
       "      <td>0.243674</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.751766</td>\n",
       "      <td>-0.199042</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.435498</td>\n",
       "      <td>0.388439</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.321976</td>\n",
       "      <td>-0.718640</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.768068</td>\n",
       "      <td>-0.391255</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         X1        X2  y\n",
       "0  0.854774  0.243674  0\n",
       "1  0.751766 -0.199042  1\n",
       "2  0.435498  0.388439  0\n",
       "3  0.321976 -0.718640  1\n",
       "4  0.768068 -0.391255  1"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df11.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "be1e3e1a-cf48-43d6-a347-15fad2c73652",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_df11 = df11[[\"X1\", \"X2\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "2ceca965-dac6-4941-8b84-fa474fe089c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_df11 = df11.y.ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "541e23e3-98a5-4e10-9afd-6a498f19c0d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_df11, X_test_df11, y_train_df11, y_test_df11 = train_test_split(X_df11, y_df11, test_size=0.25, random_state=42, stratify=y_df11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "06011c67-6501-41f4-88c7-ee49aecdb52e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# KNeighborsClassifier() with K=3\n",
    "knn3 = KNeighborsClassifier(n_neighbors=3)\n",
    "# KNeighborsClassifier() with K=4\n",
    "knn4 = KNeighborsClassifier(n_neighbors=4)\n",
    "# KNeighborsClassifier() with K=5\n",
    "knn5 = KNeighborsClassifier(n_neighbors=5)\n",
    "# DecisionTreeClassifier() with max_depth=3\n",
    "dt3 = DecisionTreeClassifier(max_depth=3, random_state=132)\n",
    "# DecisionTreeClassifier() with max_depth=4\n",
    "dt4 = DecisionTreeClassifier(max_depth=4, random_state=132)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "019e1621-a130-4fad-b786-fcda7c2dc039",
   "metadata": {},
   "outputs": [],
   "source": [
    "voting_clf = VotingClassifier(\n",
    "    estimators=[('knn3', knn3), ('knn4', knn4), ('knn5', knn5),\n",
    "               ('dt3', dt3), ('dt4', dt4)],\n",
    "    voting='hard'  # or 'soft' if you want to average predicted probabilities\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "6f98fda0-01ee-4ed4-bb15-a16ba468e048",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>VotingClassifier(estimators=[(&#x27;knn3&#x27;, KNeighborsClassifier(n_neighbors=3)),\n",
       "                             (&#x27;knn4&#x27;, KNeighborsClassifier(n_neighbors=4)),\n",
       "                             (&#x27;knn5&#x27;, KNeighborsClassifier()),\n",
       "                             (&#x27;dt3&#x27;,\n",
       "                              DecisionTreeClassifier(max_depth=3,\n",
       "                                                     random_state=132)),\n",
       "                             (&#x27;dt4&#x27;,\n",
       "                              DecisionTreeClassifier(max_depth=4,\n",
       "                                                     random_state=132))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">VotingClassifier</label><div class=\"sk-toggleable__content\"><pre>VotingClassifier(estimators=[(&#x27;knn3&#x27;, KNeighborsClassifier(n_neighbors=3)),\n",
       "                             (&#x27;knn4&#x27;, KNeighborsClassifier(n_neighbors=4)),\n",
       "                             (&#x27;knn5&#x27;, KNeighborsClassifier()),\n",
       "                             (&#x27;dt3&#x27;,\n",
       "                              DecisionTreeClassifier(max_depth=3,\n",
       "                                                     random_state=132)),\n",
       "                             (&#x27;dt4&#x27;,\n",
       "                              DecisionTreeClassifier(max_depth=4,\n",
       "                                                     random_state=132))])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>knn3</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier(n_neighbors=3)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>knn4</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier(n_neighbors=4)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>knn5</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier()</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>dt3</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=3, random_state=132)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>dt4</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=4, random_state=132)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "VotingClassifier(estimators=[('knn3', KNeighborsClassifier(n_neighbors=3)),\n",
       "                             ('knn4', KNeighborsClassifier(n_neighbors=4)),\n",
       "                             ('knn5', KNeighborsClassifier()),\n",
       "                             ('dt3',\n",
       "                              DecisionTreeClassifier(max_depth=3,\n",
       "                                                     random_state=132)),\n",
       "                             ('dt4',\n",
       "                              DecisionTreeClassifier(max_depth=4,\n",
       "                                                     random_state=132))])"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "voting_clf.fit(X_train_df11, y_train_df11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "4a8a5d5c-11b4-46c6-ba62-5d918ba948fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_df11 = voting_clf.predict(X_test_df11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "ff682079-939f-4a22-8be1-d3079f244856",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_df11 = accuracy_score(y_test_df11, y_pred_df11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "d2ef5387-76ef-4839-a699-918c233517e8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.96"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_df11"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "57cb53b9-ce7c-47a5-b4d0-b8d328c82e61",
   "metadata": {},
   "outputs": [],
   "source": [
    "f1_df11 = f1_score(y_test_df11, y_pred_df11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "6cbd5b30-5051-46d5-b38c-186e37f596fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9606299212598426"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_df11"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "abadc14f-111e-4d25-83a2-d7efb4fd48e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "precision_df11 = precision_score(y_test_df11, y_pred_df11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "c8accb6e-c7c7-4601-9568-85b721686e1f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9384615384615385"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision_df11"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "5e16af46-b679-441f-80c4-c314c6b9b23f",
   "metadata": {},
   "outputs": [],
   "source": [
    "recall_df11 = recall_score(y_test_df11, y_pred_df11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "cdf28566-2c15-45ee-b77c-aa5cef6ed8a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9838709677419355"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_df11"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "4c7596a0-192c-4087-9152-1cc0669e6553",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "knn3:\n",
      "  Precision = 0.9242\n",
      "  Recall    = 0.9839\n",
      "\n",
      "knn4:\n",
      "  Precision = 0.9531\n",
      "  Recall    = 0.9839\n",
      "\n",
      "knn5:\n",
      "  Precision = 0.9385\n",
      "  Recall    = 0.9839\n",
      "\n",
      "dt3:\n",
      "  Precision = 0.9219\n",
      "  Recall    = 0.9516\n",
      "\n",
      "dt4:\n",
      "  Precision = 0.9219\n",
      "  Recall    = 0.9516\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for name, vc in voting_clf.named_estimators_.items():\n",
    "    y_pred = vc.predict(X_test_df11)\n",
    "    precision = precision_score(y_test_df11, y_pred)\n",
    "    recall = recall_score(y_test_df11, y_pred)\n",
    "    print(f\"{name}:\")\n",
    "    print(f\"  Precision = {precision:.4f}\")\n",
    "    print(f\"  Recall    = {recall:.4f}\\n\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b9864ea-bd50-48f7-b09c-dde2512157c9",
   "metadata": {},
   "source": [
    "### Loading Dataset 13\n",
    "### Import Modules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "189d6bcf-0205-400c-a5fc-855e5681d647",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import VotingRegressor\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "c42f6a15-b49e-4d8a-9fb5-f9b442d478cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Voting Regressor\n",
    "df13 = pd.read_csv(\"APMM10-reg-dataset13.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "c28981c7-f734-46d6-9f58-2e94019379b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 6 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   X1      1000 non-null   float64\n",
      " 1   X2      1000 non-null   float64\n",
      " 2   X3      1000 non-null   float64\n",
      " 3   X4      1000 non-null   float64\n",
      " 4   X5      1000 non-null   float64\n",
      " 5   y       1000 non-null   float64\n",
      "dtypes: float64(6)\n",
      "memory usage: 47.0 KB\n"
     ]
    }
   ],
   "source": [
    "df13.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "a63b05f4-4dc2-488e-a8c0-7c52f011076a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>X1</th>\n",
       "      <th>X2</th>\n",
       "      <th>X3</th>\n",
       "      <th>X4</th>\n",
       "      <th>X5</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.778877</td>\n",
       "      <td>0.227707</td>\n",
       "      <td>-0.217250</td>\n",
       "      <td>0.117715</td>\n",
       "      <td>-0.416022</td>\n",
       "      <td>353.628836</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.984763</td>\n",
       "      <td>-1.000755</td>\n",
       "      <td>0.128196</td>\n",
       "      <td>0.588786</td>\n",
       "      <td>-1.975529</td>\n",
       "      <td>152.872646</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.729922</td>\n",
       "      <td>0.609208</td>\n",
       "      <td>1.186181</td>\n",
       "      <td>-0.594182</td>\n",
       "      <td>-0.627713</td>\n",
       "      <td>350.769222</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.777258</td>\n",
       "      <td>-1.014609</td>\n",
       "      <td>-0.970274</td>\n",
       "      <td>0.524534</td>\n",
       "      <td>-1.580636</td>\n",
       "      <td>236.262018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.046403</td>\n",
       "      <td>-0.406842</td>\n",
       "      <td>-0.736017</td>\n",
       "      <td>0.204004</td>\n",
       "      <td>-0.038934</td>\n",
       "      <td>433.694223</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         X1        X2        X3        X4        X5           y\n",
       "0 -0.778877  0.227707 -0.217250  0.117715 -0.416022  353.628836\n",
       "1 -0.984763 -1.000755  0.128196  0.588786 -1.975529  152.872646\n",
       "2 -0.729922  0.609208  1.186181 -0.594182 -0.627713  350.769222\n",
       "3  0.777258 -1.014609 -0.970274  0.524534 -1.580636  236.262018\n",
       "4  2.046403 -0.406842 -0.736017  0.204004 -0.038934  433.694223"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df13.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "7955f03d-7d9c-459b-b101-39fa2b38548f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# LinearRegression()\n",
    "lr = LinearRegression()\n",
    "# KNeighborsRegressor() with K=5\n",
    "knr5 = KNeighborsRegressor(n_neighbors=5)\n",
    "# DecisionTreeRegressor() with max_depth=5\n",
    "dtr5 = DecisionTreeRegressor(max_depth=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "e3f92165-9a37-4987-80e2-033561f050b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "voting_reg = VotingRegressor(\n",
    "        estimators=[('lr', lr), ('knr5', knr5), ('dtr5', dtr5)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "e93ad973-f2c7-4122-93fd-20f36c8c7041",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_df13 = df13[[\"X1\", \"X2\", \"X3\", \"X4\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "8761e54a-d255-40a6-a1ee-82ac71a02e71",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_df13 =df13.y.ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "1ac8b551-7b68-4bb4-8fc6-14c196c4d769",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_df13, X_test_df13, y_train_df13, y_test_df13 = train_test_split(X_df13, y_df13, test_size=0.25, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "3324c316-2188-4144-b2cd-bed5ebaf52b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>VotingRegressor(estimators=[(&#x27;lr&#x27;, LinearRegression()),\n",
       "                            (&#x27;knr5&#x27;, KNeighborsRegressor()),\n",
       "                            (&#x27;dtr5&#x27;, DecisionTreeRegressor(max_depth=5))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">VotingRegressor</label><div class=\"sk-toggleable__content\"><pre>VotingRegressor(estimators=[(&#x27;lr&#x27;, LinearRegression()),\n",
       "                            (&#x27;knr5&#x27;, KNeighborsRegressor()),\n",
       "                            (&#x27;dtr5&#x27;, DecisionTreeRegressor(max_depth=5))])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>lr</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>knr5</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsRegressor</label><div class=\"sk-toggleable__content\"><pre>KNeighborsRegressor()</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>dtr5</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" ><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeRegressor</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeRegressor(max_depth=5)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "VotingRegressor(estimators=[('lr', LinearRegression()),\n",
       "                            ('knr5', KNeighborsRegressor()),\n",
       "                            ('dtr5', DecisionTreeRegressor(max_depth=5))])"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "voting_reg.fit(X_train_df13, y_train_df13)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "bcdb3bcf-1593-435a-8876-937bd27c9e15",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_df13 = voting_reg.predict(X_test_df13)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "aa88e6a0-d3b9-47e8-abe6-940198d5b78b",
   "metadata": {},
   "outputs": [],
   "source": [
    "rsme_df13 = np.sqrt(mean_squared_error(y_test_df13, y_pred_df13))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "9352f01e-c1e7-4316-9104-16dcdb4c3c7a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q5: RSMS for Dataset 13: 114.2505\n"
     ]
    }
   ],
   "source": [
    "print(f\"Q5: RSMS for Dataset 13: {rsme_df13:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "2525cd30-c7b8-4d58-add2-040087f637fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "lr:\n",
      "  RSMS = 112.2723\n",
      "knr5:\n",
      "  RSMS = 123.6419\n",
      "dtr5:\n",
      "  RSMS = 121.8650\n"
     ]
    }
   ],
   "source": [
    "for name, vc in voting_reg.named_estimators_.items():\n",
    "    y_pred = vc.predict(X_test_df13)\n",
    "    rsms = np.sqrt(mean_squared_error(y_test_df13, y_pred))\n",
    "\n",
    "    \n",
    "    print(f\"{name}:\")\n",
    "    print(f\"  RSMS = {rsms:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e45423c8-9c27-46ed-bd34-993b68e7e480",
   "metadata": {},
   "source": [
    "### Loading Dataset 14\n",
    "### Import Modules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "23f5a9ad-7758-4c8f-8246-59ddb476d89f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "dfbecdb2-4c6e-4e70-9e21-0c5927c27c0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Perform a train-test split using a 70-30 split and ensure stratify=yes.\n",
    "# For all functions where random_state is a parameter, use random_state=42."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "a9c74551-7261-4a84-ae3a-18ce30a0a115",
   "metadata": {},
   "outputs": [],
   "source": [
    "df14 = pd.read_csv(\"APMM10-clf-dataset14.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "31fa25f1-2e1d-4b3c-b559-c44d2a448ccb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 7 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   X1      1000 non-null   float64\n",
      " 1   X2      1000 non-null   float64\n",
      " 2   X3      1000 non-null   float64\n",
      " 3   X4      1000 non-null   float64\n",
      " 4   X5      1000 non-null   float64\n",
      " 5   X6      1000 non-null   float64\n",
      " 6   y       1000 non-null   int64  \n",
      "dtypes: float64(6), int64(1)\n",
      "memory usage: 54.8 KB\n"
     ]
    }
   ],
   "source": [
    "df14.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "5d628e8b-f48b-4699-b56c-f7d871b98268",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>X1</th>\n",
       "      <th>X2</th>\n",
       "      <th>X3</th>\n",
       "      <th>X4</th>\n",
       "      <th>X5</th>\n",
       "      <th>X6</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.424860</td>\n",
       "      <td>-0.579713</td>\n",
       "      <td>-1.337395</td>\n",
       "      <td>0.362592</td>\n",
       "      <td>0.117978</td>\n",
       "      <td>-1.414848</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.540444</td>\n",
       "      <td>0.844824</td>\n",
       "      <td>1.151952</td>\n",
       "      <td>-1.201070</td>\n",
       "      <td>0.341755</td>\n",
       "      <td>-0.841155</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.871295</td>\n",
       "      <td>-3.570698</td>\n",
       "      <td>-1.304408</td>\n",
       "      <td>1.673602</td>\n",
       "      <td>-0.026235</td>\n",
       "      <td>3.035470</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.234720</td>\n",
       "      <td>-3.148500</td>\n",
       "      <td>-2.922666</td>\n",
       "      <td>1.387281</td>\n",
       "      <td>1.525623</td>\n",
       "      <td>2.645610</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.774487</td>\n",
       "      <td>1.968075</td>\n",
       "      <td>0.657095</td>\n",
       "      <td>0.739276</td>\n",
       "      <td>-1.782984</td>\n",
       "      <td>-0.678737</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         X1        X2        X3        X4        X5        X6  y\n",
       "0 -0.424860 -0.579713 -1.337395  0.362592  0.117978 -1.414848  0\n",
       "1  0.540444  0.844824  1.151952 -1.201070  0.341755 -0.841155  0\n",
       "2  1.871295 -3.570698 -1.304408  1.673602 -0.026235  3.035470  0\n",
       "3  0.234720 -3.148500 -2.922666  1.387281  1.525623  2.645610  0\n",
       "4 -1.774487  1.968075  0.657095  0.739276 -1.782984 -0.678737  0"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df14.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "5c6834c2-3aa6-49b2-8d99-8eb7cc8b50f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_df14 = df14[[\"X1\", \"X2\", \"X3\", \"X4\", \"X5\", \"X6\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "53aebd0a-f3e6-4084-bcbe-c5872c39e3f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_df14 = df14.y.ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "814e0ba1-9527-40a0-ac40-9f98e3ee4509",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_df14, X_test_df14, y_train_df14, y_test_df14 = train_test_split(X_df14, y_df14, test_size=0.30, random_state=42, stratify=y_df14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "1698a9a6-3993-46cf-81db-bb3aa20358ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Random Tree Classifier\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "635d67f9-9c99-4368-944a-a6c7d203f089",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" checked><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 118,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf.fit(X_train_df14, y_train_df14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "56f7b3de-c17f-4dba-9e75-3fbad4856cdd",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_df14 = rf.predict(X_test_df14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "3415c2ca-480b-4462-a889-1e213a62e114",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy = accuracy_score(y_test_df14, y_pred_df14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "51fcd17f-99ff-47a8-90f7-1e3b1db161d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.86"
      ]
     },
     "execution_count": 121,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "e26ce2c2-8d73-455d-9e00-059629a242d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "f1s_df14 = f1_score(y_test_df14, y_pred_df14, average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "5a87f35e-c8ad-4781-abb1-53434565eb28",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8531489759024178"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1s_df14"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "f14c05d4-f3d4-4907-9dd8-020c5e8f805a",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y_test_df14, y_pred_df14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "3de153d5-2d2e-41d1-853a-320f558474ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[208,  10,   2],\n",
       "       [ 13,  34,   1],\n",
       "       [ 14,   2,  16]])"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "b4b83465-ca98-49fe-a0fc-59b364eb1578",
   "metadata": {},
   "outputs": [],
   "source": [
    "df14_test = pd.DataFrame(y_test_df14, columns=[\"test\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "2c68a531-bd73-4859-b033-a27ed3dd462b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df14_test[\"pred\"] = y_pred_df14"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "b88d6148-3c8c-4d17-bb42-3d784cf22873",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>test</th>\n",
       "      <th>pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>295</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>296</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>300 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     test  pred\n",
       "0       0     0\n",
       "1       0     0\n",
       "2       0     0\n",
       "3       2     2\n",
       "4       0     0\n",
       "..    ...   ...\n",
       "295     0     0\n",
       "296     1     1\n",
       "297     0     1\n",
       "298     0     0\n",
       "299     0     0\n",
       "\n",
       "[300 rows x 2 columns]"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df14_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "8148847e-f23e-4807-bbb0-f7a5ad350f49",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "test    208\n",
       "pred    208\n",
       "dtype: int64"
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# TP for Label 0\n",
    "df14_test.loc[(df14_test.test == 0) & (df14_test.pred == 0)].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "bb0474fb-209e-4036-8b70-1c8c6c353845",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "test    12\n",
       "pred    12\n",
       "dtype: int64"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# FN for Label 0\n",
    "df14_test.loc[(df14_test.test == 0) & ~(df14_test.pred == 0)].count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f193fde-e129-4450-ab6a-d323667c3dc4",
   "metadata": {},
   "source": [
    "### Load Dataset 17\n",
    "### Import Modules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "b7e17caa-cbb7-4fb8-b7c6-1f1a6c2b7abd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "dedda34b-8982-45c1-b225-a8ebd7e0f641",
   "metadata": {},
   "outputs": [],
   "source": [
    "df17 = pd.read_csv(\"APMM10-clf-dataset17.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "8e1e292a-e484-42de-b47f-b6b52a4f5f4d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 5 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   X1      1000 non-null   float64\n",
      " 1   X2      1000 non-null   float64\n",
      " 2   X3      1000 non-null   float64\n",
      " 3   X4      1000 non-null   float64\n",
      " 4   y       1000 non-null   int64  \n",
      "dtypes: float64(4), int64(1)\n",
      "memory usage: 39.2 KB\n"
     ]
    }
   ],
   "source": [
    "df17.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "id": "1837b060-2e7c-49cf-8fca-a665a53a6e80",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>X1</th>\n",
       "      <th>X2</th>\n",
       "      <th>X3</th>\n",
       "      <th>X4</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.194848</td>\n",
       "      <td>1.138331</td>\n",
       "      <td>-1.482079</td>\n",
       "      <td>1.025358</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.290131</td>\n",
       "      <td>0.317681</td>\n",
       "      <td>-0.996519</td>\n",
       "      <td>0.477009</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.522737</td>\n",
       "      <td>-0.121434</td>\n",
       "      <td>0.765888</td>\n",
       "      <td>-0.308384</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.738138</td>\n",
       "      <td>-0.188065</td>\n",
       "      <td>-1.195371</td>\n",
       "      <td>0.302161</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.828884</td>\n",
       "      <td>-0.289261</td>\n",
       "      <td>-1.049151</td>\n",
       "      <td>0.206271</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         X1        X2        X3        X4  y\n",
       "0 -1.194848  1.138331 -1.482079  1.025358  0\n",
       "1  0.290131  0.317681 -0.996519  0.477009  0\n",
       "2 -0.522737 -0.121434  0.765888 -0.308384  0\n",
       "3  1.738138 -0.188065 -1.195371  0.302161  0\n",
       "4  1.828884 -0.289261 -1.049151  0.206271  0"
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df17.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "id": "640db451-ae49-4fbb-a1e0-a80d6f9fa705",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_df17 = df17[[\"X1\", \"X2\", \"X3\", \"X4\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "76bca96e-a781-4cb6-9a54-1828d7374afb",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_df17 = df17.y.ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "id": "0f3c4267-a883-4898-b9e6-541d0409dbf2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Boosting\n",
    "# Build a boosting classifier ensemble using the AdaBoostClassifier() method. \n",
    "# Use a DecisionTreeClassifier with max_depth=1 and random_state=42 as the “stump”. \n",
    "# Initialize the AdaBoostClassifier() with 60 estimators, a learning_rate of 1.0, and a random_state of 42. \n",
    "# 70-30 Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "id": "f5fc6434-2b49-424f-bb29-a20cc28a3226",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_df17, X_test_df17, y_train_df17, y_test_df17 = train_test_split(X_df17, y_df17, test_size=0.30, random_state=42, stratify=y_df17)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "id": "748e5589-dc5e-494d-9c23-31cd524316a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "stump = DecisionTreeClassifier(max_depth=1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "id": "dcead853-d674-465d-b295-5960b86a1a58",
   "metadata": {},
   "outputs": [],
   "source": [
    "ada = AdaBoostClassifier(estimator=stump, n_estimators=60, learning_rate=1.0, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "0949ddf2-de52-443d-8fc9-4d8d975516d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,\n",
       "                                                    random_state=42),\n",
       "                   n_estimators=60, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-12\" type=\"checkbox\" ><label for=\"sk-estimator-id-12\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">AdaBoostClassifier</label><div class=\"sk-toggleable__content\"><pre>AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,\n",
       "                                                    random_state=42),\n",
       "                   n_estimators=60, random_state=42)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" ><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=1, random_state=42)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" ><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=1, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,\n",
       "                                                    random_state=42),\n",
       "                   n_estimators=60, random_state=42)"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ada.fit(X_train_df17, y_train_df17)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "c65baffc-133e-4b2a-a2b4-8b4514d120bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_df17 = ada.predict(X_test_df17)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "24aa9ed8-c5d8-463c-a468-7e137218d1df",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_df17 = accuracy_score(y_test_df17, y_pred_df17)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "ced6248c-4cac-4458-b01e-b0dd7d0a66fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9533333333333334"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_df17"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "8ebfe8c9-266c-40ad-95e8-5bcfacc7c07a",
   "metadata": {},
   "outputs": [],
   "source": [
    "f1s_df17 = f1_score(y_test_df17, y_pred_df17, average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "96cf7ed5-9778-4134-8c25-b48df2f9070e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9510906862745098"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1s_df17"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "f62fc731-78ef-4165-8967-c44dffd847a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier(max_depth=1, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-15\" type=\"checkbox\" checked><label for=\"sk-estimator-id-15\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=1, random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier(max_depth=1, random_state=42)"
      ]
     },
     "execution_count": 147,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stump.fit(X_train_df17, y_train_df17)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "1e15bfdd-13a1-4e51-80c1-9f3dbf5246f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_stump = stump.predict(X_test_df17)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "18e3c69a-41dd-429d-b31d-0d4ce87a917e",
   "metadata": {},
   "outputs": [],
   "source": [
    "precision_stump = precision_score(y_test_df17, y_pred_stump, average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "id": "55457478-ed40-40bc-bebd-d830b1a11671",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9267809228954739"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision_stump"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "70eee89a-85a4-4853-b35c-cf239ab32df5",
   "metadata": {},
   "outputs": [],
   "source": [
    "recall_stump = recall_score(y_test_df17, y_pred_stump, average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "af6b61c8-c995-44e8-9db4-f09ebcac1a96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9233333333333333"
      ]
     },
     "execution_count": 152,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_stump"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56fc46c0-7def-4372-938c-43e3bd2776b6",
   "metadata": {},
   "source": [
    "### Loading Dataset 18\n",
    "### Import Modules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "421e6281-cf67-48ff-90e2-e0fec4dde22c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import StackingClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "98de77aa-e6e8-4a0e-8f4a-32b7c79e3cb6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df18 = pd.read_csv(\"APMM10-clf-dataset18.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "3205338d-ac78-4780-a994-684d55a0379d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 7 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   X1      1000 non-null   float64\n",
      " 1   X2      1000 non-null   float64\n",
      " 2   X3      1000 non-null   float64\n",
      " 3   X4      1000 non-null   float64\n",
      " 4   X5      1000 non-null   float64\n",
      " 5   X6      1000 non-null   float64\n",
      " 6   y       1000 non-null   int64  \n",
      "dtypes: float64(6), int64(1)\n",
      "memory usage: 54.8 KB\n"
     ]
    }
   ],
   "source": [
    "df18.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "ab1da0f2-72f7-4555-b6d8-81d0ceada689",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>X1</th>\n",
       "      <th>X2</th>\n",
       "      <th>X3</th>\n",
       "      <th>X4</th>\n",
       "      <th>X5</th>\n",
       "      <th>X6</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.700765</td>\n",
       "      <td>-1.958677</td>\n",
       "      <td>0.224340</td>\n",
       "      <td>-0.145380</td>\n",
       "      <td>-1.309757</td>\n",
       "      <td>-0.401478</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.201960</td>\n",
       "      <td>0.749327</td>\n",
       "      <td>0.849612</td>\n",
       "      <td>0.656523</td>\n",
       "      <td>1.458497</td>\n",
       "      <td>1.049780</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2.011720</td>\n",
       "      <td>-1.810551</td>\n",
       "      <td>0.434770</td>\n",
       "      <td>0.615294</td>\n",
       "      <td>0.947045</td>\n",
       "      <td>0.241003</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.580108</td>\n",
       "      <td>-2.191164</td>\n",
       "      <td>0.159463</td>\n",
       "      <td>0.613825</td>\n",
       "      <td>0.306174</td>\n",
       "      <td>0.428269</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.232211</td>\n",
       "      <td>2.800755</td>\n",
       "      <td>0.765878</td>\n",
       "      <td>0.617917</td>\n",
       "      <td>2.319216</td>\n",
       "      <td>1.294300</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         X1        X2        X3        X4        X5        X6  y\n",
       "0  0.700765 -1.958677  0.224340 -0.145380 -1.309757 -0.401478  0\n",
       "1 -0.201960  0.749327  0.849612  0.656523  1.458497  1.049780  1\n",
       "2  2.011720 -1.810551  0.434770  0.615294  0.947045  0.241003  1\n",
       "3  1.580108 -2.191164  0.159463  0.613825  0.306174  0.428269  0\n",
       "4 -1.232211  2.800755  0.765878  0.617917  2.319216  1.294300  1"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df18.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "dde35ff6-18a4-40b6-9186-e9cf1703050d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_df18 = df18[[\"X1\", \"X2\", \"X3\", \"X4\", \"X5\", \"X6\" ]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "0c2c743e-6098-4326-bd36-5211eeba407d",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_df18 = df18.y.ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "d46fe361-5dc2-4417-afe9-fb0856f09433",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_df18, X_test_df18, y_train_df18, y_test_df18 = train_test_split(X_df18, y_df18, test_size=0.30, random_state=42, stratify=y_df18)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "c971522a-55e3-41ec-9b0f-3291a74af4b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# LogisticRegression() with a max_iter limit of 1000,\n",
    "lr = LogisticRegression(max_iter=1000)\n",
    "\n",
    "# DecisionTreeClassifier() with a max_depth of 3 (and random_state=42),\n",
    "dtc3 = DecisionTreeClassifier(max_depth=3, random_state=42)\n",
    "\n",
    "# DecisionTreeClassifier() with a max_depth of 5 (and random_state=42)\n",
    "dtc5 = DecisionTreeClassifier(max_depth=5, random_state=42)\n",
    "\n",
    "# KNeighborsClassifier() with k=5, and\n",
    "knn5 = KNeighborsClassifier(n_neighbors=5)\n",
    "\n",
    "# KNeighborsClassifier() with k=7\n",
    "knn7 = KNeighborsClassifier(n_neighbors=7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "cbf0cb60-ef92-4b96-991b-78a1438109d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "stack = StackingClassifier(\n",
    "            estimators=[('dtc3', dtc3), ('dtc5', dtc5), ('knn5', knn5), ('knn7', knn7), ('lr', lr)],\n",
    "            final_estimator=LogisticRegression(),\n",
    "            cv=5\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "fdf3f934-1ff7-4c51-9d59-a423bbf11dac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-6 {color: black;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-6\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>StackingClassifier(cv=5,\n",
       "                   estimators=[(&#x27;dtc3&#x27;,\n",
       "                                DecisionTreeClassifier(max_depth=3,\n",
       "                                                       random_state=42)),\n",
       "                               (&#x27;dtc5&#x27;,\n",
       "                                DecisionTreeClassifier(max_depth=5,\n",
       "                                                       random_state=42)),\n",
       "                               (&#x27;knn5&#x27;, KNeighborsClassifier()),\n",
       "                               (&#x27;knn7&#x27;, KNeighborsClassifier(n_neighbors=7)),\n",
       "                               (&#x27;lr&#x27;, LogisticRegression(max_iter=1000))],\n",
       "                   final_estimator=LogisticRegression())</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-16\" type=\"checkbox\" ><label for=\"sk-estimator-id-16\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StackingClassifier</label><div class=\"sk-toggleable__content\"><pre>StackingClassifier(cv=5,\n",
       "                   estimators=[(&#x27;dtc3&#x27;,\n",
       "                                DecisionTreeClassifier(max_depth=3,\n",
       "                                                       random_state=42)),\n",
       "                               (&#x27;dtc5&#x27;,\n",
       "                                DecisionTreeClassifier(max_depth=5,\n",
       "                                                       random_state=42)),\n",
       "                               (&#x27;knn5&#x27;, KNeighborsClassifier()),\n",
       "                               (&#x27;knn7&#x27;, KNeighborsClassifier(n_neighbors=7)),\n",
       "                               (&#x27;lr&#x27;, LogisticRegression(max_iter=1000))],\n",
       "                   final_estimator=LogisticRegression())</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>dtc3</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-17\" type=\"checkbox\" ><label for=\"sk-estimator-id-17\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=3, random_state=42)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>dtc5</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-18\" type=\"checkbox\" ><label for=\"sk-estimator-id-18\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=5, random_state=42)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>knn5</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-19\" type=\"checkbox\" ><label for=\"sk-estimator-id-19\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier()</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>knn7</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-20\" type=\"checkbox\" ><label for=\"sk-estimator-id-20\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier(n_neighbors=7)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>lr</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-21\" type=\"checkbox\" ><label for=\"sk-estimator-id-21\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(max_iter=1000)</pre></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>final_estimator</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-22\" type=\"checkbox\" ><label for=\"sk-estimator-id-22\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "StackingClassifier(cv=5,\n",
       "                   estimators=[('dtc3',\n",
       "                                DecisionTreeClassifier(max_depth=3,\n",
       "                                                       random_state=42)),\n",
       "                               ('dtc5',\n",
       "                                DecisionTreeClassifier(max_depth=5,\n",
       "                                                       random_state=42)),\n",
       "                               ('knn5', KNeighborsClassifier()),\n",
       "                               ('knn7', KNeighborsClassifier(n_neighbors=7)),\n",
       "                               ('lr', LogisticRegression(max_iter=1000))],\n",
       "                   final_estimator=LogisticRegression())"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# stack.fit(X_train5, y_train5)\n",
    "stack.fit(X_train_df18, y_train_df18)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "88694980-3025-4082-9a04-c547c105558b",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_df18 = stack.predict(X_test_df18)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "cdc4a829-cf1b-45e6-92af-f424cd44f5fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_df18 = accuracy_score(y_test_df18, y_pred_df18)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "ec8c57aa-2ea0-4214-a5c6-550759d1f657",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.94"
      ]
     },
     "execution_count": 165,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_df18"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "5140b678-ca19-4a2f-90a4-3020206437db",
   "metadata": {},
   "outputs": [],
   "source": [
    "f1score_df18 = f1_score(y_test_df18, y_pred_df18, average=\"weighted\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "94084b07-6af3-4b21-8d83-0b208d00418e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9401283796926668"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1score_df18"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "ac40ff55-4e03-44f1-9ea4-8878430d7bcf",
   "metadata": {},
   "outputs": [],
   "source": [
    "precision_df18 = precision_score(y_test_df18, y_pred_df18, average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "25abd1cd-3888-4d61-8255-b4575169f090",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9403467144448211"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision_df18"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "0a6e6739-ee67-4346-87a5-7030db361e04",
   "metadata": {},
   "outputs": [],
   "source": [
    "recall_df18 = recall_score(y_test_df18, y_pred_df18, average='weighted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "498e7be9-79b4-4edd-b87f-cd811c7599f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.94"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_df18"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "523654d3-59c8-4d36-a3be-24d838063f71",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dtc3:\n",
      "        Precision Score = 0.923522\n",
      "        Recall Score = 0.923333\n",
      "dtc5:\n",
      "        Precision Score = 0.930331\n",
      "        Recall Score = 0.930000\n",
      "knn5:\n",
      "        Precision Score = 0.940000\n",
      "        Recall Score = 0.940000\n",
      "knn7:\n",
      "        Precision Score = 0.943895\n",
      "        Recall Score = 0.943333\n",
      "lr:\n",
      "        Precision Score = 0.900000\n",
      "        Recall Score = 0.900000\n"
     ]
    }
   ],
   "source": [
    "for name, vc in stack.named_estimators_.items():\n",
    "    y_pred = vc.predict(X_test_df18)\n",
    "    \n",
    "    # Calculate precision\n",
    "    precision = precision_score(y_test_df18, y_pred, average='weighted')  # or 'macro'/'micro' if you prefer\n",
    "    recall = recall_score(y_test_df18, y_pred, average='weighted')\n",
    "    print(f\"{name}:\")\n",
    "    print(f\"        Precision Score = {precision:.6f}\")\n",
    "    print(f\"        Recall Score = {recall:.6f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4af8f395-99a0-4d3b-be9f-4ff881155fd3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db590079-20d2-4669-a48a-f009da5b1d4c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c52c7e4a-2fb7-48e6-9148-c4d51ab3188d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
