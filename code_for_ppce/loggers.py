import logging

##loggers several quantities

# results_coalitions
logger = logging.getLogger("results")
logger.setLevel(logging.INFO)

app_handler = logging.FileHandler("results.log")
app_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
app_handler.setFormatter(app_formatter)
logger.addHandler(app_handler)

#results shapley
svlogger = logging.getLogger("shapley")
svlogger.setLevel(logging.INFO)

svhandler = logging.FileHandler("shapley.log")
svformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
svhandler.setFormatter(svformatter)
svlogger.addHandler(svhandler)


# results ppce
ppcelogger = logging.getLogger("ppce")
ppcelogger.setLevel(logging.INFO)

ppcehandler = logging.FileHandler("ppce.log")
ppceformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ppcehandler.setFormatter(ppceformatter)
ppcelogger.addHandler(ppcehandler)


# results metrics
metricslogger = logging.getLogger("metrics")
metricslogger.setLevel(logging.INFO)

metricshandler = logging.FileHandler("metrics.log")
metricsformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
metricshandler.setFormatter(metricsformatter)
metricslogger.addHandler(metricshandler)

#mean_std metrics
mean_stdlogger = logging.getLogger("mean_std")
mean_stdlogger.setLevel(logging.INFO)

mean_stdhandler = logging.FileHandler("mean_std.log")
mean_stdformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
mean_stdhandler.setFormatter(mean_stdformatter)
mean_stdlogger.addHandler(mean_stdhandler)



# # Example Usage
# app_logger.info("Application started successfully.")
# user_logger.info("User logged in with ID 12345.")