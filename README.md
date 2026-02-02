sa# \# Heart\_attack\_pred

# 

# \&gt; One-sentence elevator pitch of what your model / app does.

# 

# \## Description

# \[2–3 sentences explaining the business problem, the data you use, and the ML goal.]

# 

# Example:  

# We predict the likelihood of customer churn for a Portuguese banking institution using demographic and behavioural features.  

# The end-product is a REST API served behind a lightweight React dashboard that allows marketers to upload new customer lists and receive churn probabilities in real time.

# 

# \## Dataset

# | Source | Link | Size | License |

# |--------|------|------|---------|

# | \[Bank Marketing UCI] | https://archive.ics.uci.edu/ml/datasets/bank+marketing | 45 211 rows × 21 cols | CC BY 4.0 |

# 

# (Include a short “Data dictionary” subsection here if you wish.)

# 

# \## 7-Week Roadmap

# | Week | Milestone | Deliverable |

# |------|-----------|-------------|

# | 1 | Scraping \& EDA | Jupyter notebook with visualisations + data report |

# | 2 | Pre-processing \& Feature Engineering | Clean pipeline + engineered features stored in `data/processed/` |

# | 3 | Modelling (Gradient Boosting) \& MLflow | Best model registered in MLflow, `metrics.txt` |

# | 4 | API (FastAPI) | `/predict` endpoint + automated tests |

# | 5 | Frontend (React) | Single-page app that consumes the API |

# | 6 | Containerisation (Docker) | `docker-compose up` spins up full stack |

# | 7 | Deployment \& Review | App live on AWS / Render + final presentation |

# 

# \## Quick Start

# ```bash

# git clone https://github.com/YOUR\_GITHUB\_HANDLE/YOUR\_REPO\_NAME.git

# cd YOUR\_REPO\_NAME

# python -m venv venv \&\& source venv/bin/activate

# pip install -r requirements.txt

