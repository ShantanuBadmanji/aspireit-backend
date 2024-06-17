import express from "express";

import {
    getJobposts,
    getJobpostsById,
    createJobPost,
    updateJobPost,
    deleteJobPost,
    applyJobPost,
    getApplicantsByJobPostId,
} from "../controllers/jobPostFunction.js";

const router = express.Router();

router.route("/").get(getJobposts).post(createJobPost);

router
    .route("/:id")
    .get(getJobpostsById)
    .put(updateJobPost)
    .delete(deleteJobPost);

router
    .route("/:id/applicants")
    .get(getApplicantsByJobPostId)
    .post(applyJobPost);

export default router;
