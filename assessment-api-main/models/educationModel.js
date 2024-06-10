import mongoose from "mongoose";

const educationSchema = new mongoose.Schema({
    course:{
        type: String,
        enum: ['B.Tech', 'M.Tech', 'MBA', 'BBA', 'BCA', 'MCA', 'B.Com', 'M.Com'],
        default: 'B.Tech',
    },
    institute:{
        type: String,
        required: true
    },
    cgpa:{
        type: Number,
        required: true
    },
    passing_year:{
        type: Number,
        required: true
    },
    candidate_id:{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Candidate',
        required: true
    }
}, {timestamps: true});

const educationModel = mongoose.model('Education', educationSchema);

export default educationModel;