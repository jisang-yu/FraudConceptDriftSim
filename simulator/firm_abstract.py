from mesa import Agent
from abc import ABCMeta, abstractmethod
import numpy as np

class AbstractFirm(Agent, metaclass=ABCMeta):
    def __init__(self, gvkey_year, reporting_model, fraudster):
        """
        Abstract class for firms, which can either be genuine or fraudulent.
        :param gvkey_year:           the (unique) firm ID (gvkey-year unique identifier)
        :param reporting_model:   the transaction mod   el that is used, instance of mesa.Model
        :param fraudster:           boolean whether firm is genuine or fraudulent
        """
        # call super init from mesa agent
        super().__init__(gvkey_year, reporting_model)

        # copy parameters from model
        self.params = self.model.parameters

        # internal random state (different for every firm)
        self.random_state = np.random.RandomState(self.model.random_state.randint(0, np.iinfo(np.int32).max))

        # each firm has to say if it's a fraudster or not
        self.fraudster = int(fraudster)

        # pick reporting file id
        self.file_id = None # picked with first file report


        # variable for whether a reporting is currently being processed
        self.active = False

        # current reporting properties (at, cogs, lt, ni, ppegt)
        self.curr_sec = None
        self.at = None
        self.cogs = None
        self.lt = None
        self.ni = None
        self.ppegt = None

        self.local_datetime = None
        self.curr_auth_step = 0
        self.curr_report_cancelled = False
        self.curr_report_success = False

        # variable tells us whether the firm wants to stay after current quarter
        self.stay = True

    def step(self):
        """
        This is called in each simulation step (i.e., one quarter).
        Each individual firm/fraudster decides whether to report or not.
        """
        file_report = self.decide_filing_report()


        if file_report:

            # if this is the first file report, we assign a file ID
            if self.file_id is None:
                self.file_id = self.initialize_file_id()

            # set the agent to active
            self.active = True

            # pick current SEC
            self.curr_sec = self.get_curr_sec()

            # pick a current amount
            self.at = self.get_curr_at()
            self.cogs = self.get_curr_cogs()
            self.lt = self.get_curr_lt()
            self.ni = self.get_curr_ni()
            self.ppegt = self.get_curr_ppegt()

            # process current reporting
            self.curr_report_success = self.model.process_report(self)

            # if necessary post_process the reporting
            self.post_process_report()

        else:

            # set to inactive (important for report logs)
            self.active = False
            self.curr_sec = None

            self.at = None
            self.cogs = None
            self.lt = None
            self.ni = None
            self.ppegt = None

            self.local_datetime = None

    def request_report(self):
        self.model.authorize_report(self)

    def post_process_report(self):
        """
        Optional updates after transaction;
        e.g. decide whether to stay or update satisfaction
        :return: """
        pass

    @abstractmethod
    def give_authentication(self):
        """
        Authenticate self if requested by the payment processing platform.
        Return can e.g. be quality of authentication or boolean.
        If no authentication is given, this usually returns None.
        :return:
        """
        pass

    @abstractmethod
    def get_curr_sec(self):
        pass

    @abstractmethod
    def get_curr_at(self):
        pass

    @abstractmethod
    def get_curr_cogs(self):
        pass

    @abstractmethod
    def get_curr_lt(self):
        pass

    @abstractmethod
    def get_curr_in(self):
        pass

    @abstractmethod
    def get_curr_ppegt(self):
        pass

    @abstractmethod
    def decide_filing_report(self):
        """
        Decide whether to file report or not, given the current time step
        :return:    Boolean indicating whether to make transaction or not
        """
        pass

    @abstractmethod
    def initialize_file_id(self):
        """
        Select file ID number (unique ID) for firm
        :return:    file ID number
        """
        pass