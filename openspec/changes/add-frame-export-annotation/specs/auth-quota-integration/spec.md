# Specification: Authentication and Quota Integration

## ADDED Requirements

### Requirement: Validate user authentication for video processing

The system SHALL validate user authentication for all video upload, annotation, and export operations via JWT tokens from the existing iDoctor auth service.

#### Scenario: Authenticated user accesses video annotation features

**Given** a user has a valid JWT token from the auth service
**When** the user uploads a video or initiates an export
**Then** the backend shall validate the JWT signature and expiration
**And** extract the user ID from the token claims
**And** allow the operation to proceed
**And** associate all data with the authenticated user ID

#### Scenario: Unauthenticated user attempts to use the service

**Given** a user visits the demo without authentication
**When** the user attempts to upload a video
**Then** the system shall redirect to the login page
**Or** display an authentication modal if using embedded auth
**And** prevent any video processing until authenticated

#### Scenario: Expired token is rejected

**Given** a user's JWT token has expired
**When** the user attempts any protected operation
**Then** the backend shall return a 401 Unauthorized error
**And** the frontend shall display "Session expired, please log in again"
**And** redirect to login or trigger token refresh flow
**And** not process the requested operation

---

### Requirement: Enforce quota limits on video processing

The system SHALL enforce quota limits on video processing based on user subscription tiers, tracked through the existing iDoctor quota service.

#### Scenario: User within quota limits processes video

**Given** a user has 100 minutes of video processing quota remaining
**When** the user uploads a 2-minute video
**Then** the backend shall check the quota service before processing
**And** confirm 100 minutes available
**And** allow the video to be processed
**And** deduct 2 minutes from the user's quota after successful processing
**And** update the quota service with new balance (98 minutes)

#### Scenario: User exceeding quota is blocked

**Given** a user has 1 minute of quota remaining
**When** the user attempts to upload a 5-minute video
**Then** the backend shall reject the upload before processing
**And** return a 403 Forbidden error with message "Quota exceeded"
**And** the frontend shall display:
- "You have insufficient quota (1 min remaining, 5 min needed)"
- Link to upgrade subscription or purchase additional quota
**And** not deduct any quota for failed request

#### Scenario: Export operations consume quota

**Given** a user exports a 60-second video at 10 FPS (600 frames)
**When** the export is initiated
**Then** the backend shall calculate quota cost (e.g., 1 export operation = 5 minutes)
**And** verify sufficient quota before processing
**And** deduct the quota cost only after successful export
**And** refund quota if export fails due to system error (not user error)

---

### Requirement: Track usage metrics for quota calculation

The system SHALL track usage metrics and report resource usage to the quota service for proper billing and limit enforcement.

#### Scenario: Video processing duration is tracked

**Given** a user uploads a 3-minute 45-second video
**When** the video is processed by SAM2VideoPredictor
**Then** the system shall record:
- Video duration: 225 seconds
- Processing start and end timestamps
- Actual GPU processing time
**And** report to quota service: "video_processing" metric, 225 seconds
**And** round up to nearest minute for quota deduction (4 minutes)

#### Scenario: Export operations are tracked separately

**Given** a user performs an export
**Then** the system shall track:
- Export operation count
- Number of frames processed
- Total processing time
**And** report metrics to quota service as "export_operation" event
**And** apply appropriate quota cost based on operation complexity

#### Scenario: Failed operations do not consume quota

**Given** a user's video upload fails during processing (corrupt file)
**When** the backend detects the error
**Then** the system shall not deduct any quota
**And** log the failure for debugging
**And** notify the user of the error without quota penalty

---

### Requirement: Provide quota visibility to users

The system SHALL provide quota visibility to users, displaying current quota status and consumption to enable effective usage management.

#### Scenario: User views quota status in UI

**Given** an authenticated user with 50 minutes of quota remaining
**When** the user is on the annotation page
**Then** the UI shall display:
- "Quota remaining: 50 minutes"
- Visual progress bar showing quota usage
- Link to quota details page
**And** update quota display after each operation
**And** warn when quota drops below 20%

#### Scenario: Pre-operation quota check shows cost

**Given** a user is about to upload a 10-minute video
**When** the file is selected
**Then** the system shall display:
- "This video will use ~10 minutes of quota"
- "Current balance: 50 minutes → 40 minutes after processing"
**And** require confirmation before uploading
**And** block upload if quota would be exceeded

#### Scenario: Quota exhaustion prevents new operations

**Given** a user has 0 minutes of quota remaining
**When** the user tries to upload a new video
**Then** the system shall block the upload immediately
**And** display a message:
- "Quota exhausted. Upgrade or purchase additional quota to continue."
- Clear call-to-action button linking to subscription management
**And** still allow viewing and exporting previously processed videos (cached results)

---

## MODIFIED Requirements

_None - this is a new integration, no existing requirements are modified_

## REMOVED Requirements

_None - no requirements are being removed_
