show databases;

use titanic_db;
show tables;
use iris_db;
show tables;
use telco_churn;
show tables;

describe measurements;
describe species;
SELECT * 
FROM measurements
	JOIN species
		USING (species_id);
        

-- customers, contract_types, internet_service_types, payment_types 
describe customers;
describe contract_types;
describe internet_service_types;
describe payment_types;

SELECT *
FROM customers
	JOIN contract_types
		USING (contract_type_id)
	JOIN internet_service_types
		USING (internet_service_type_id)
	JOIN payment_types
		USING (payment_type_id)
limit 5;