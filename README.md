# CLEAR-Automated Analysis Generation

### I. Unstructured data pipeline

1. Pull Dataset from unified data source (data class)
2. Process PDF documents and data
3. entry extraction
4. entry classification to framework, with kept specific tags
  4.1 First level only: BERT-like

### II. Analysis generation

1. Context analysis: Pillar-level (Web-based, last year) + KPIs on keys pre-crisis vulnerabilities
2. Narrative of latest shocks from 2 months (web-based, from start date, specific prompt), extract number of poeple in need. USING DATAMINR
3. displacement with all subpillars
4. Humanitarian Access with all subpillars
5. Sector-wise analysis: one sector, one pillar. not more. Including overview, with sources for each, with reliability score
  2.1 humanitarian conditions & Severity (structured extraction, severity)  
    2.2 Impact & priority needs (qualitative + Number of people affected, systems and service facilities severly damaged/destroyed, with what in particular, then priority needs as a list with a KPI for each)
6. KPIs (including reasoning on decision-making for each number given, with ranges)
  3.1 Number of people displaced  
    3.2 Number of people in need  
    3.3 Number of people dead, injured, missing
7. From generated anlayses, list of risks  
  7.1 Number of People Exposed  
    7.2 Short-term proba  
    7.3 Mid-term proba  
    7.4 Early Warning Indicators  



### HTML with key indicators and small pop-ups when needed

### Report downloadable in word format

later
- recommended actions
- forecasting of numbers
- scenarios
- crisis status
