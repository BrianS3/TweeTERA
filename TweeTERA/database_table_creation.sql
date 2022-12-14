DROP TABLE IF EXISTS TWEET_TEXT;
CREATE TABLE TWEET_TEXT (
TWEET_ID VARCHAR(50) NOT NULL,
AUTHOR_ID VARCHAR(50),
CREATED TEXT,
SEARCH_TERM TEXT,
TIDY_TWEET TEXT,
LEMM TEXT,
HASHTAGS TEXT,
OVERALL_EMO TEXT,
PRIMARY KEY (TWEET_ID)
);

DROP TABLE IF EXISTS USERS;
CREATE TABLE USERS (
AUTHOR_ID VARCHAR(50) NOT NULL,
CREATED_AT DATE,
LOCATION TEXT,
FOLLOWERS_COUNT INT,
FOLLOWING_COUNT INT,
LISTED_COUNT INT,
TWEET_COUNT INT,
VERIFIED TEXT,
PRIMARY KEY(AUTHOR_ID)
);

DROP TABLE IF EXISTS REGIONS;
CREATE TABLE REGIONS (
REG_ID INT NOT NULL AUTO_INCREMENT,
REGION TEXT,
PRIMARY KEY(REG_ID)
);

INSERT INTO REGIONS (REGION)
VALUES ('NORTHEAST'),
('MIDWEST'),
('WEST'),
('SOUTH');

DROP TABLE IF EXISTS DIVISIONS;
CREATE TABLE DIVISIONS (
DIV_ID INT NOT NULL AUTO_INCREMENT,
DIVISION TEXT,
REG_ID INT,
PRIMARY KEY(DIV_ID)
);

INSERT INTO DIVISIONS (DIVISION, REG_ID)
VALUES ('NEW ENGLAND', (SELECT REG_ID FROM REGIONS WHERE REGION = 'NORTHEAST')),
('MIDDLE ATLANTIC', (SELECT REG_ID FROM REGIONS WHERE REGION = 'NORTHEAST')),
('EAST NORTH CENTRAL', (SELECT REG_ID FROM REGIONS WHERE REGION = 'MIDWEST')),
('WEST NORTH CENTRAL', (SELECT REG_ID FROM REGIONS WHERE REGION = 'MIDWEST')),
('MOUNTAIN', (SELECT REG_ID FROM REGIONS WHERE REGION = 'WEST')),
('PACIFIC', (SELECT REG_ID FROM REGIONS WHERE REGION = 'WEST')),
('SOUTH ATLANTIC', (SELECT REG_ID FROM REGIONS WHERE REGION = 'SOUTH')),
('EAST SOUTH CENTRAL', (SELECT REG_ID FROM REGIONS WHERE REGION = 'SOUTH')),
('WEST SOUTH CENTRAL', (SELECT REG_ID FROM REGIONS WHERE REGION = 'SOUTH'));

DROP TABLE IF EXISTS US_STATES;
CREATE TABLE US_STATES (
STATE_ID INT NOT NULL AUTO_INCREMENT,
STATE TEXT,
STATE_ABBR TEXT,
DIV_ID INT,
PRIMARY KEY(STATE_ID)
);

INSERT INTO US_STATES (STATE_ABBR,STATE,DIV_ID)
VALUES ('AL','ALABAMA', (SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST SOUTH CENTRAL')),
('AK','ALASKA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'PACIFIC')),
('AZ','ARIZONA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN')),
('AR','ARKANSAS',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST SOUTH CENTRAL')),
('CA','CALIFORNIA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'PACIFIC')),
('CO','COLORADO',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN')),
('CT','CONNECTICUT',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'NEW ENGLAND')),
('DE','DELAWARE',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('DC','DISTRICT OF COLUMBIA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('FL','FLORIDA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('GA','GEORGIA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('HI','HAWAII',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'PACIFIC')),
('ID','IDAHO',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN')),
('IL','ILLINOIS',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST NORTH CENTRAL')),
('IN','INDIANA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST NORTH CENTRAL')),
('IA','IOWA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST NORTH CENTRAL')),
('KS','KANSAS',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST NORTH CENTRAL')),
('KY','KENTUCKY',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST SOUTH CENTRAL')),
('LA','LOUISIANA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST SOUTH CENTRAL')),
('ME','MAINE',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'NEW ENGLAND')),
('MD','MARYLAND',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('MA','MASSACHUSETTS',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'NEW ENGLAND')),
('MI','MICHIGAN',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST NORTH CENTRAL')),
('MN','MINNESOTA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST NORTH CENTRAL')),
('MS','MISSISSIPPI',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST SOUTH CENTRAL')),
('MO','MISSOURI',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST NORTH CENTRAL')),
('MT','MONTANA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN')),
('NE','NEBRASKA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST NORTH CENTRAL')),
('NV','NEVADA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN')),
('NH','NEW HAMPSHIRE',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'NEW ENGLAND')),
('NJ','NEW JERSEY',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MIDDLE ATLANTIC')),
('NM','NEW MEXICO',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN')),
('NY','NEW YORK',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MIDDLE ATLANTIC')),
('NC','NORTH CAROLINA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('ND','NORTH DAKOTA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST NORTH CENTRAL')),
('OH','OHIO',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST NORTH CENTRAL')),
('OK','OKLAHOMA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST SOUTH CENTRAL')),
('OR','OREGON',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'PACIFIC')),
('PA','PENNSYLVANIA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MIDDLE ATLANTIC')),
('RI','RHODE ISLAND',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'NEW ENGLAND')),
('SC','SOUTH CAROLINA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('SD','SOUTH DAKOTA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST NORTH CENTRAL')),
('TN','TENNESSEE',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST SOUTH CENTRAL')),
('TX','TEXAS',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'WEST SOUTH CENTRAL')),
('UT','UTAH',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN')),
('VT','VERMONT',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'NEW ENGLAND')),
('VA','VIRGINIA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('WA','WASHINGTON',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'PACIFIC')),
('WV','WEST VIRGINIA',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'SOUTH ATLANTIC')),
('WI','WISCONSIN',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'EAST NORTH CENTRAL')),
('WY','WYOMING',(SELECT DIV_ID FROM DIVISIONS WHERE DIVISION = 'MOUNTAIN'));

DROP TABLE IF EXISTS AUTHOR_LOCATION;
CREATE TABLE AUTHOR_LOCATION(
AUTHOR_ID VARCHAR(50),
STATE_ID INT,
PRIMARY KEY(AUTHOR_ID)
);