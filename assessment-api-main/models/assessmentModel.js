import mongoose from "mongoose";

const assessmentSchema = new mongoose.Schema({
    questions:{
        type: [mongoose.Schema.Types.ObjectId],
        ref: 'Question',
        default: []
    },
    role:{
        type: String,
        enum: ['frontend', 'backend', 'fullstack', 'analyst', 'finance', 'marketing', 'hr'],
        default: 'fullstack'
    },
    duration:{
        type: Number,
        required: true
    }
},{timestamps: true});

const assessmentModel = mongoose.model('Assessment', assessmentSchema);

export default assessmentModel;