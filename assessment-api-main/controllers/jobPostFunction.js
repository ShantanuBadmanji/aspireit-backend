import JobPost from "../models/jobPostModel.js";
import candidateModel from "../models/candidateModel.js";

const getJobposts = async (req, res) => {
    try {
        const jobPosts = await JobPost.find();
        res.json(jobPosts);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

const getJobpostsById = async (req, res) => {
    try {
        const jobPost = await JobPost.findById(req.params.id);
        if (jobPost) {
            res.json(jobPost);
        } else {
            res.status(404).json({ message: "Job Post not found" });
        }
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

const createJobPost = async (req, res) => {
    const {
        title,
        company,
        location,
        remote,
        description,
        screeningQuestions,
        spent,
        dailyBudget,
    } = req.body;

    try {
        const jobPost = new JobPost({
            title,
            company,
            location,
            remote,
            description,
            screeningQuestions,
            spent,
            dailyBudget,
        });

        const createdJobPost = await jobPost.save();
        res.status(201).json(createdJobPost);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
};

const updateJobPost = async (req, res) => {
    const {
        title,
        company,
        location,
        remote,
        description,
        screeningQuestions,
        spent,
        dailyBudget,
    } = req.body;

    try {
        const jobPost = await JobPost.findById(req.params.id);
        if (jobPost) {
            jobPost.title = title;
            jobPost.company = company;
            jobPost.location = location;
            jobPost.remote = remote;
            jobPost.description = description;
            jobPost.screeningQuestions = screeningQuestions;
            jobPost.spent = spent;
            jobPost.dailyBudget = dailyBudget;
            const updatedJobPost = await jobPost.save();
            res.status(201).json(updatedJobPost);
        } else {
            res.status(404).json({ message: "Job Post not found" });
        }
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
};

const deleteJobPost = async (req, res) => {
    try {
        const jobPost = await JobPost.findById(req.params.id);

        if (!jobPost) {
            return res.status(404).json({ message: "Job post not found" });
        }

        await jobPost.deleteOne();

        res.json({ message: "Job post removed" });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Server error" });
    }
};
const applyJobPost = async (req, res) => {
    const { jobApplicantId } = req.body;
    try {
        const updatedJobPost = await JobPost.findByIdAndUpdate(
            req.params.id,
            { $push: { applicants: jobApplicantId } },
            { new: true, runValidators: true }
        );

        if (!updatedJobPost) {
            return res.status(404).json({ message: "Job Post not found" });
        }

        res.json(updatedJobPost);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
};
const getApplicantsByJobPostId = async (req, res) => {
    try {
        const jobPost = await JobPost.findById(req.params.id).populate('applicants');
        if (!jobPost) {
            return res.status(404).json({ message: "Job Post not found" });
        }

        res.json(jobPost.applicants);
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: error.message });
    }
};
export {
    getJobposts,
    getJobpostsById,
    createJobPost,
    updateJobPost,
    deleteJobPost,
    applyJobPost,
    getApplicantsByJobPostId,
};
